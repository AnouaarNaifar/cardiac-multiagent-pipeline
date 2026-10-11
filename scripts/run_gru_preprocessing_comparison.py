"""Run GRU preprocessing experiments and compare them with the CNN."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import wfdb
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg_embedding import (
    DIAGNOSTIC_SUPERCLASSES,
    ECGEmbeddingNet,
    ECGGRUNet,
    PTBXLDataset,
    load_scp_superclass_map,
    scp_codes_to_multihot,
    train_embedding_model,
)

DATA_ROOT = ROOT / "kaggle" / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUTPUT_DIR = ROOT / "results" / "ecg_embedding"
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3


class PreprocessedPTBXLDataset(Dataset):
    """PTB-XL dataset with one explicit raw-signal transform per experiment."""

    def __init__(self, database_csv, scp_statements_csv, dataset_dir, preprocess_fn):
        self.dataset_dir = Path(dataset_dir)
        self.df = __import__("pandas").read_csv(database_csv).dropna(
            subset=["scp_codes"]
        ).reset_index(drop=True)
        self.code_to_superclass = load_scp_superclass_map(scp_statements_csv)
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.df)

    def strat_fold_indices(self, train_folds=range(1, 9), val_fold=9, test_fold=10):
        folds = self.df["strat_fold"].to_numpy()
        return (
            np.nonzero(np.isin(folds, list(train_folds)))[0],
            np.nonzero(folds == val_fold)[0],
            np.nonzero(folds == test_fold)[0],
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = self.dataset_dir / row["filename_lr"]
        raw_signal, metadata = wfdb.rdsamp(str(record_path))
        processed_signal = self.preprocess_fn(raw_signal, float(metadata["fs"]))
        label = scp_codes_to_multihot(row["scp_codes"], self.code_to_superclass)
        return torch.from_numpy(processed_signal.T), torch.from_numpy(label)


def load_checkpoint(model_class, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_class(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def evaluate(model, dataset, test_indices):
    loader = DataLoader(Subset(dataset, test_indices.tolist()), batch_size=64, shuffle=False)
    criterion = torch.nn.BCEWithLogitsLoss()
    all_probabilities, all_labels = [], []
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for signals, labels in loader:
            logits = model(signals)
            loss_sum += criterion(logits, labels).item() * signals.size(0)
            all_probabilities.append(torch.sigmoid(logits).numpy())
            all_labels.append(labels.numpy())

    labels = np.concatenate(all_labels).astype(np.int32)
    predictions = (np.concatenate(all_probabilities) >= 0.5).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "test_loss": loss_sum / len(labels),
        "exact_match": np.all(predictions == labels, axis=1).mean(),
    }


def train_variant(name, preprocess_fn):
    dataset = PreprocessedPTBXLDataset(
        DATA_ROOT / "ptbxl_database.csv",
        DATA_ROOT / "scp_statements.csv",
        DATA_ROOT,
        preprocess_fn,
    )
    train_indices, val_indices, _ = dataset.strat_fold_indices()
    checkpoint_path = OUTPUT_DIR / f"{name}.best.pt"
    if checkpoint_path.exists():
        print(f"{name}: checkpoint already exists, skipping training", flush=True)
        return

    model = ECGGRUNet()
    print(f"Starting {name}: train={len(train_indices)} val={len(val_indices)}", flush=True)
    train_embedding_model(
        dataset=Subset(dataset, train_indices.tolist()),
        val_dataset=Subset(dataset, val_indices.tolist()),
        model=model,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path=str(OUTPUT_DIR / name),
        epoch_log_path=str(OUTPUT_DIR / f"epoch_log_{name}.csv"),
    )
    print(f"Finished {name}", flush=True)


def make_row(label, model, checkpoint, dataset, test_indices):
    metrics = evaluate(model, dataset, test_indices)
    row = {
        "model": label,
        "validation_loss_best": min(checkpoint["history"]["val_loss"]),
        "test_loss": metrics["test_loss"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "exact_match": metrics["exact_match"],
    }
    for index, class_name in enumerate(DIAGNOSTIC_SUPERCLASSES):
        row[f"precision_{class_name}"] = metrics["precision"][index]
        row[f"recall_{class_name}"] = metrics["recall"][index]
        row[f"f1_{class_name}"] = metrics["f1"][index]
    return row


def write_results(rows):
    csv_path = OUTPUT_DIR / "cnn_gru_preprocessing_comparison.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = OUTPUT_DIR / "cnn_gru_preprocessing_comparison.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# CNN vs GRU preprocessing comparison\n\n")
        handle.write("Test set: PTB-XL `strat_fold` fold 10, threshold 0.5.\n\n")
        handle.write("| Model | F1 CD | F1 HYP | F1 MI | F1 NORM | F1 STTC | F1 macro | Precision macro | Recall macro | Best val loss | Test loss |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['f1_CD']:.3f} | {row['f1_HYP']:.3f} | "
                f"{row['f1_MI']:.3f} | {row['f1_NORM']:.3f} | {row['f1_STTC']:.3f} | "
                f"{row['f1_macro']:.3f} | {row['precision_macro']:.3f} | "
                f"{row['recall_macro']:.3f} | {row['validation_loss_best']:.3f} | "
                f"{row['test_loss']:.3f} |\n"
            )
    print(f"Results written to {markdown_path}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_dataset = PTBXLDataset(
        DATA_ROOT / "ptbxl_database.csv",
        DATA_ROOT / "scp_statements.csv",
        DATA_ROOT,
        resolution="lr",
    )
    _, _, test_indices = base_dataset.strat_fold_indices()

    # Temporarily disabled: baseline/notch and wavelet GRU experiments.
    preprocess_variants = []

    rows = []
    cnn, cnn_checkpoint = load_checkpoint(
        ECGEmbeddingNet, OUTPUT_DIR / "embedding_stratfold.best.pt"
    )
    rows.append(make_row("CNN", cnn, cnn_checkpoint, base_dataset, test_indices))

    gru_norm, gru_norm_checkpoint = load_checkpoint(
        ECGGRUNet, OUTPUT_DIR / "gru_stratfold.best.pt"
    )
    rows.append(make_row("GRU + normalization", gru_norm, gru_norm_checkpoint, base_dataset, test_indices))

    for name, preprocess_fn in preprocess_variants:
        model, checkpoint = load_checkpoint(ECGGRUNet, OUTPUT_DIR / f"{name}.best.pt")
        dataset = PreprocessedPTBXLDataset(
            DATA_ROOT / "ptbxl_database.csv",
            DATA_ROOT / "scp_statements.csv",
            DATA_ROOT,
            preprocess_fn,
        )
        label = "GRU + baseline + notch" if "wavelet" not in name else "GRU + baseline + notch + wavelet"
        rows.append(make_row(label, model, checkpoint, dataset, test_indices))

    write_results(rows)


if __name__ == "__main__":
    main()
