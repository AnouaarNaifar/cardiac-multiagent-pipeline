"""
ECG learned embedding - architecture and data pipeline
========================================================
Phase E of the ECG agent: a learned, fixed-size representation of the raw
12-lead signal, to complement the hand-crafted features in ecg_agent.py
(heart rate, RR intervals, QRS duration, ...) with information those
statistics cannot capture (P-wave/T-wave shape, ST-segment morphology, etc.).

This module defines the model architectures and the PTB-XL data pipeline
needed to train them. Two architecture families are kept:
  - ECGEmbeddingNet, a 1D-CNN ending in global average pooling: this is the
    FINAL production model, wired into ecg_agent.py (checkpoint
    embedding_stratfold.best.pt) - it has the best calibrated test F1 macro
    and the best HYP F1.
  - ECGGRUNet, a GRU reading the signal as a temporal sequence: the retained
    recurrent baseline (addresses the supervisor's "think about temporal
    recurrence" note), NOT a replacement for the CNN on these metrics; plus
    ECGGRU2LayerDropoutNet, a deeper experimental variant that was tried and
    NOT retained.
Both families accept a VARIABLE-LENGTH input - the same weights run on 100 Hz
(1000 samples) or 500 Hz (5000 samples) PTB-XL records unchanged, with no fixed
input size imposed: the CNN via global average pooling, the GRU by reading the
sequence step by step. The recurrent-vs-CNN and preprocessing comparisons live
in results/ecg_embedding/ (cnn_gru_preprocessing_comparison.md,
rnn_threshold_validation.md): conclusion - the CNN remains the final model, the
simple GRU is the cleanest recurrent baseline, and per-lead z-score
normalization only (no baseline/notch/wavelet) is used. Training itself
(train_embedding_model) is implemented here but not run automatically.

Task used to train the embedding: PTB-XL's standard 5-class diagnostic
superclass benchmark (NORM, MI, STTC, HYP, CD), derived from each record's
"scp_codes" via "scp_statements.csv". The embedding is the model's
penultimate-layer output (before the classification head), so it inherits
whatever diagnostic-relevant structure the classifier learns, without being
just the 5 raw class scores.

Additional dependency beyond the rest of the project: PyTorch.
    pip install torch
(A GPU via CUDA is used automatically if available, but is not required.)
"""

import ast
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Note: extra signal-preprocessing variants (baseline removal + powerline notch,
# optionally + wavelet denoising) were evaluated for the GRU and dropped in favor
# of per-lead z-score normalization only - see
# results/ecg_embedding/cnn_gru_preprocessing_comparison.md.

try:
    import wfdb
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Install it with: pip install wfdb"
    ) from exc


DIAGNOSTIC_SUPERCLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]


# ----------------------------------------------------------------------
# 1. Label mapping (scp_codes -> multi-hot diagnostic superclass vector)
# ----------------------------------------------------------------------
def load_scp_superclass_map(scp_statements_csv):
    """
    Build a mapping from SCP-ECG code (e.g. "NORM", "IMI") to one of the 5
    PTB-XL diagnostic superclasses (see DIAGNOSTIC_SUPERCLASSES), keeping
    only codes flagged as diagnostically relevant in scp_statements.csv.

    Inputs:
        scp_statements_csv: path to PTB-XL's scp_statements.csv.

    Output:
        dict mapping scp_code (str) -> diagnostic superclass (str).
    """
    df = pd.read_csv(scp_statements_csv, index_col=0)
    df = df[df["diagnostic"] == 1] #diagnostic == 1 est un filtre qui sépare "vrais diagnostics" (utilisables pour la classification) des "métadonnées techniques sur le tracé" (non pertinentes ici) — les deux types de codes coexistent dans le même fichier, cette ligne isole seulement le premier groupe.
    return df["diagnostic_class"].dropna().to_dict()


def scp_codes_to_multihot(scp_codes_str, code_to_superclass):
    """
    Convert a record's raw "scp_codes" string (e.g.
    "{'NORM': 100.0, 'LVOLT': 0.0}") into a multi-hot label vector over
    DIAGNOSTIC_SUPERCLASSES. A superclass is set to 1 if at least one of its
    codes has a nonzero likelihood for this record.

    Inputs:
        scp_codes_str: raw string value from the PTB-XL database's
            "scp_codes" column.
        code_to_superclass: mapping produced by load_scp_superclass_map.

    Output:
        1D numpy float32 array of length len(DIAGNOSTIC_SUPERCLASSES).
    """
    codes = ast.literal_eval(scp_codes_str)
    label = np.zeros(len(DIAGNOSTIC_SUPERCLASSES), dtype=np.float32)
    for code, likelihood in codes.items():
        superclass = code_to_superclass.get(code)
        if superclass in DIAGNOSTIC_SUPERCLASSES and likelihood > 0:
            label[DIAGNOSTIC_SUPERCLASSES.index(superclass)] = 1.0
    return label


# ----------------------------------------------------------------------
# 2. PTB-XL dataset (12-lead signal -> multi-hot diagnostic superclass label)
# ----------------------------------------------------------------------
class PTBXLDataset(Dataset):
    """
    PyTorch Dataset over PTB-XL: each item is a (signal, label) pair where
    signal is the raw 12-lead ECG (per-lead z-score normalized) and label is
    the multi-hot diagnostic superclass vector for that record.

    Inputs (constructor):
        database_csv: path to PTB-XL's ptbxl_database.csv.
        scp_statements_csv: path to PTB-XL's scp_statements.csv.
        dataset_dir: root directory containing the "records100"/"records500"
            folders (the same directory that holds ptbxl_database.csv).
        resolution: "lr" (100 Hz, filename_lr column) or "hr" (500 Hz,
            filename_hr column). "lr" is faster to train on and is the
            default.

    Output (per __getitem__ call):
        (signal, label) tuple: signal is a float32 torch tensor of shape
        (12, n_samples); label is a float32 torch tensor of shape
        (len(DIAGNOSTIC_SUPERCLASSES),).
    """

    def __init__(self, database_csv, scp_statements_csv, dataset_dir, resolution="lr"):
        self.dataset_dir = Path(dataset_dir)
        self.df = pd.read_csv(database_csv).dropna(subset=["scp_codes"]).reset_index(drop=True)
        self.code_to_superclass = load_scp_superclass_map(scp_statements_csv)
        self.filename_col = "filename_lr" if resolution == "lr" else "filename_hr"

    def strat_fold_indices(self, train_folds=range(1, 9), val_fold=9, test_fold=10):
        """
        Split this dataset's record indices using PTB-XL's official
        "strat_fold" column - a patient-stratified 10-fold assignment
        (verified: no patient_id ever spans more than one fold), so a split
        built from it is guaranteed leakage-free at the patient level, unlike
        a plain random per-record split (which can put some of a patient's
        other records in the training set while one of their records sits in
        the held-out eval set).

        Inputs:
            train_folds: iterable of strat_fold values used for training
                (default: PTB-XL's documented standard, folds 1-8).
            val_fold: strat_fold value used for validation (default: 9).
            test_fold: strat_fold value used as the held-out test/eval set
                (default: 10).

        Output:
            (train_indices, val_indices, test_indices) - each a 1D numpy
            array of integer positions into this dataset (suitable for
            torch.utils.data.Subset).
        """
        folds = self.df["strat_fold"].to_numpy()
        train_indices = np.nonzero(np.isin(folds, list(train_folds)))[0]
        val_indices = np.nonzero(folds == val_fold)[0]
        test_indices = np.nonzero(folds == test_fold)[0]
        return train_indices, val_indices, test_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_path = self.dataset_dir / row[self.filename_col]
        sig, _meta = wfdb.rdsamp(str(record_path))

        sig = sig.astype(np.float32).T  # (n_leads, n_samples)
        sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)

        label = scp_codes_to_multihot(row["scp_codes"], self.code_to_superclass)
        return torch.from_numpy(sig), torch.from_numpy(label)


# ----------------------------------------------------------------------
# 3. Model architecture (small 1D-CNN with an embedding bottleneck)
# ----------------------------------------------------------------------
class ECGEmbeddingNet(nn.Module):
    """
    Small 1D convolutional network over 12-lead ECG signals. The
    convolutional stack ends in global average pooling, so it accepts any
    input length (works with both 100 Hz and 500 Hz PTB-XL records without
    changes). The embedding is the pooled feature vector, taken before the
    final linear classification head.

    Inputs (constructor):
        n_leads: number of ECG leads in the input (12 for PTB-XL).
        embedding_dim: size of the learned embedding vector.
        n_classes: number of output classes (len(DIAGNOSTIC_SUPERCLASSES)).
    """

    def __init__(self, n_leads=12, embedding_dim=64, n_classes=len(DIAGNOSTIC_SUPERCLASSES)):
        super().__init__()
        self.n_leads = n_leads
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.conv_stack = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def embed(self, x):
        #embed(x) : retourne juste le vecteur de features (avant la couche de classification) — c'est ça, l'embedding qu'on veut réutiliser dans l'agent ECG
        """
        Compute the embedding vector for a batch of signals.

        Inputs:
            x: float tensor of shape (batch, n_leads, n_samples).

        Output:
            float tensor of shape (batch, embedding_dim).
        """
        features = self.conv_stack(x)
        return self.pool(features).squeeze(-1)

    def forward(self, x):
        #forward(x) : fait le calcul complet jusqu'à la classification — utilisé uniquement pendant l'entraînement
        """
        Full forward pass: embedding followed by the classification head.

        Inputs:
            x: float tensor of shape (batch, n_leads, n_samples).

        Output:
            float tensor of shape (batch, n_classes) - raw logits (use
            sigmoid for per-class probabilities, since this is multi-label).
        """
        return self.classifier(self.embed(x))


# ----------------------------------------------------------------------
# 3b. Model architecture (GRU with an embedding bottleneck)
# ----------------------------------------------------------------------
class ECGGRUNet(nn.Module):
    """
    Retained recurrent model for multilead ECG sequences.

    The public input format stays compatible with PTBXLDataset and the CNN:
    (batch, n_leads, n_samples). Internally, the GRU reads the signal as a
    temporal sequence of 12-lead observations: (batch, n_samples, n_leads).
    The final hidden state is the learned embedding. The selected experiment
    uses this one-layer GRU with per-lead z-score normalization only.
    """

    def __init__(self, n_leads=12, embedding_dim=64, n_classes=len(DIAGNOSTIC_SUPERCLASSES)):
        super().__init__()
        self.n_leads = n_leads
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.gru = nn.GRU(
            input_size=n_leads,
            hidden_size=embedding_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def embed(self, x):
        """
        Compute the final GRU hidden state for a batch of signals.

        Input:
            x: float tensor of shape (batch, n_leads, n_samples).

        Output:
            float tensor of shape (batch, embedding_dim).
        """
        sequence = x.transpose(1, 2)
        _outputs, hidden = self.gru(sequence)
        return hidden[-1]

    def forward(self, x):
        """Return raw multilabel logits of shape (batch, n_classes)."""
        return self.classifier(self.embed(x))


class ECGGRU2LayerDropoutNet(ECGGRUNet):
    """Two-layer GRU variant with recurrent-stack dropout.

    It keeps the same input, embedding, and five-class output contract as
    ECGGRUNet, while adding capacity and dropout between the two GRU layers.
    """

    def __init__(self, n_leads=12, embedding_dim=64, n_classes=len(DIAGNOSTIC_SUPERCLASSES), dropout=0.2):
        nn.Module.__init__(self)
        self.n_leads = n_leads
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.num_layers = 2
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=n_leads,
            hidden_size=embedding_dim,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)


# ----------------------------------------------------------------------
# 4. Inference: embedding a single preprocessed signal
# ----------------------------------------------------------------------
def embed_signal(sig_multilead, model, device=None):
    """
    Compute the learned embedding for a single 12-lead ECG signal.

    Inputs:
        sig_multilead: numpy array of shape (n_leads, n_samples), already
            normalized the same way as during training (see PTBXLDataset -
            per-lead z-score).
        model: an ECGEmbeddingNet or ECGGRUNet instance (trained or not - an
            untrained model returns a valid but not diagnostically meaningful
            vector).
        device: torch device string ("cpu" or "cuda"). None (default) =
            auto-detect (CUDA if available, else CPU); pass "cpu" explicitly
            to force CPU even when a GPU is present.

    Output:
        1D numpy float32 array of length model's embedding_dim.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sig_multilead.astype(np.float32)).unsqueeze(0).to(device)
        embedding = model.embed(x)
    return embedding.cpu().numpy()[0]


def predict_diagnostic_probabilities(sig_multilead, model, device=None):
    """
    Compute per-class diagnostic-superclass probabilities for a single
    12-lead ECG signal, via the model's classification head (sigmoid over
    forward()'s raw logits, since this is a multi-label task).

    Inputs:
        sig_multilead: numpy array of shape (n_leads, n_samples), already
            normalized the same way as during training (see PTBXLDataset -
            per-lead z-score).
        model: a trained ECGEmbeddingNet or ECGGRUNet instance (see
            load_model_checkpoint).
        device: torch device string ("cpu" or "cuda"). None (default) =
            auto-detect (CUDA if available, else CPU); pass "cpu" explicitly
            to force CPU even when a GPU is present.

    Output:
        1D numpy float32 array of length len(DIAGNOSTIC_SUPERCLASSES), one
        probability per class, in DIAGNOSTIC_SUPERCLASSES order.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(sig_multilead.astype(np.float32)).unsqueeze(0).to(device)
        probabilities = torch.sigmoid(model(x))
    return probabilities.cpu().numpy()[0]


# ----------------------------------------------------------------------
# 5. Checkpoint helpers (atomic save, architecture metadata, epoch log)
# ----------------------------------------------------------------------
def _model_architecture(model):
    """Return a dict of the constructor args needed to reconstruct `model`."""
    architecture = {
        "n_leads": model.n_leads,
        "embedding_dim": model.embedding_dim,
        "n_classes": model.n_classes,
    }
    if hasattr(model, "num_layers"):
        architecture["dropout"] = model.dropout
    return architecture


def _check_architecture_match(model, saved_architecture):
    """
    Raise a clear ValueError if a checkpoint's saved architecture doesn't
    match the given model instance, instead of failing later with a cryptic
    state_dict tensor-shape mismatch.
    """
    if saved_architecture is None:
        return
    current = _model_architecture(model)
    if current != saved_architecture:
        raise ValueError(
            f"Checkpoint architecture {saved_architecture} does not match "
            f"the given model's architecture {current}."
        )


def _atomic_torch_save(obj, path):
    """
    Save `obj` (via torch.save) atomically: write to a temporary file in the
    same directory, then rename it into place with os.replace. Protects
    against a corrupted checkpoint if the process is interrupted mid-write -
    training can be stopped at any epoch, and a half-written file would
    otherwise make the checkpoint unloadable.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _append_epoch_log(log_path, epoch, train_loss, val_loss, epoch_seconds):
    """
    Append one row (epoch, train_loss, val_loss, epoch_seconds) to a CSV
    training log, writing the header first if the file doesn't exist yet.
    Kept independent of the checkpoint mechanism, so the loss curve survives
    even if a checkpoint write itself is interrupted.
    """
    log_path = Path(log_path)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["epoch", "train_loss", "val_loss", "epoch_seconds"])
        writer.writerow([epoch, train_loss, val_loss, f"{epoch_seconds:.2f}"])


# ----------------------------------------------------------------------
# 6. Training (defined here, not run automatically - see module docstring)
# ----------------------------------------------------------------------
def train_embedding_model(
    dataset,
    model,
    epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    val_fraction=0.1,
    val_dataset=None,
    device=None,
    checkpoint_path=None,
    epoch_log_path=None,
    resume_from=None,
    pos_weight=None,
    patience=None,
):
    """
    Train an ECGEmbeddingNet or ECGGRUNet on the 5-class diagnostic superclass
    task (multi-label). Not called automatically anywhere in this module - run
    it explicitly when ready to train (see module docstring).

    Checkpointing: if checkpoint_path is given, a resumable checkpoint
    (model + optimizer state, epoch number, loss history, architecture
    metadata) is written atomically after every completed epoch to
    "<checkpoint_path>.latest.pt", and separately to
    "<checkpoint_path>.best.pt" whenever validation loss improves - so the
    best model is never overwritten by a later, worse epoch. If interrupted
    (KeyboardInterrupt or any other exception mid-epoch), the ".latest.pt"
    checkpoint still reflects the last fully completed epoch, and the
    exception is re-raised so the caller knows training did not finish
    normally rather than silently returning a partial result.

    Inputs:
        dataset: a PTBXLDataset instance.
        model: an ECGEmbeddingNet or ECGGRUNet instance to train in place.
        epochs: total epoch count to reach (if resuming, training continues
            from the resumed epoch up to this total, not for `epochs` more).
        batch_size: minibatch size.
        learning_rate: Adam learning rate.
        val_fraction: fraction of the dataset held out for validation, via a
            plain random split. Ignored if val_dataset is given.
        val_dataset: optional pre-built validation dataset (e.g. a Subset
            produced from PTBXLDataset.strat_fold_indices), used instead of
            splitting `dataset` internally. Use this whenever `dataset` and
            the eventual held-out test set must both come from a patient-
            disjoint split (a random split here would not respect that,
            since it operates on records rather than patients) - `dataset`
            is then used in full for training, with no records withheld.
        device: torch device string ("cpu" or "cuda"). None (default) =
            auto-detect (CUDA if available, else CPU); pass "cpu" explicitly
            to force CPU even when a GPU is present.
        checkpoint_path: base path (without extension) for resumable
            checkpoints; if None, no checkpoint is written.
        epoch_log_path: path to a CSV file to append one row per completed
            epoch to; if None, no log is written.
        resume_from: path to a ".latest.pt"/".best.pt" checkpoint previously
            written by this function; if given, model/optimizer/epoch/
            history are restored from it before training continues.
        pos_weight: optional tensor or sequence of positive-class weights for
            BCEWithLogitsLoss. None keeps the original unweighted loss.
        patience: optional early-stopping patience. If set to an int, training
            stops as soon as the validation loss has not improved for `patience`
            consecutive epochs (the best epoch is still the one saved to
            "<checkpoint_path>.best.pt"). None (default) disables early stopping
            and trains for the full `epochs`, preserving the original behavior.

    Output:
        dict with "train_loss" and "val_loss" lists (one entry per epoch
        completed so far, including any resumed history).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if pos_weight is not None:
        pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_epoch = 0
    history = {"train_loss": [], "val_loss": []}

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        _check_architecture_match(model, checkpoint.get("architecture"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    epochs_without_improvement = 0

    if val_dataset is not None:
        train_set, val_set = dataset, val_dataset
    else:
        val_size = int(len(dataset) * val_fraction)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
    train_size, val_size = len(train_set), len(val_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()

            model.train()
            running_loss = 0.0
            for signals, labels in train_loader:
                signals, labels = signals.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(signals)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * signals.size(0)
            train_loss = running_loss / train_size

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    logits = model(signals)
                    val_running_loss += criterion(logits, labels).item() * signals.size(0)
            val_loss = val_running_loss / max(val_size, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            completed_epoch = epoch + 1

            if epoch_log_path is not None:
                _append_epoch_log(
                    epoch_log_path, completed_epoch, train_loss, val_loss, time.time() - epoch_start
                )

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if checkpoint_path is not None:
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": completed_epoch,
                    "history": history,
                    "architecture": _model_architecture(model),
                }
                _atomic_torch_save(state, f"{checkpoint_path}.latest.pt")
                if improved:
                    _atomic_torch_save(state, f"{checkpoint_path}.best.pt")

            if patience is not None and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {completed_epoch}: validation loss "
                    f"has not improved for {patience} epoch(s) (best {best_val_loss:.4f}). "
                    f"The best epoch is preserved in '.best.pt' if a checkpoint_path was given."
                )
                break
    except KeyboardInterrupt:
        completed = len(history["train_loss"])
        if checkpoint_path is not None:
            print(
                f"Training interrupted after epoch {completed} - "
                f"'{checkpoint_path}.latest.pt' reflects this last completed epoch."
            )
        else:
            print(
                f"Training interrupted after epoch {completed} - no checkpoint_path "
                "was given, so this run's progress was not saved."
            )
        raise

    return history


def save_model_checkpoint(model, path):
    """
    Save an ECGEmbeddingNet's or ECGGRUNet's weights to disk, atomically,
    along with its architecture metadata (n_leads, embedding_dim, n_classes,
    and dropout for the 2-layer GRU variant) so a later
    load_model_checkpoint can fail with a clear error instead of a cryptic
    state_dict shape mismatch if the architecture doesn't match.
    """
    state = {"model_state_dict": model.state_dict(), "architecture": _model_architecture(model)}
    _atomic_torch_save(state, path)


def load_model_checkpoint(model, path, device="cpu"):
    """
    Load previously saved weights into an ECGEmbeddingNet or ECGGRUNet
    instance, in place. Raises ValueError if the checkpoint's architecture
    metadata doesn't match the given model instance.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _check_architecture_match(model, checkpoint.get("architecture"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
