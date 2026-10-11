"""
End-to-end validation of the learned MRI pathology classifier on the 50
held-out test patients (never used in training or model selection for
YOLO, U-Net, or this classifier). Aggregates per-slice predictions to one
prediction per patient (mean probability across that patient's slices,
argmax), then compares against the real ACDC pathology-group ground truth.
"""
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mri_classifier_dataset import ACDCPathologyClassificationDataset, PATHOLOGY_CLASSES
from mri_classifier_train import PathologyClassifier

_WEIGHTS = "results/mri_classifier_training/best.pt"


def run_test_validation(output_dir="results/mri_classifier_validation"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PathologyClassifier().to(device)
    model.load_state_dict(torch.load(_WEIGHTS, map_location=device))
    model.eval()

    df = pd.read_csv("results/mri_yolo_dataset/manifest.csv")
    test_df = df[df.split == "test"]
    test_ds = ACDCPathologyClassificationDataset(test_df, augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    patient_probs, patient_true = {}, {}
    with torch.no_grad():
        for images, labels, patient_ids in test_loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu()
            for i, pid in enumerate(patient_ids):
                patient_probs.setdefault(pid, []).append(probs[i])
                patient_true[pid] = labels[i].item()

    rows = []
    for pid, probs_list in patient_probs.items():
        mean_probs = torch.stack(probs_list).mean(dim=0)
        pred_idx = int(mean_probs.argmax())
        rows.append({
            "patient_id": pid,
            "true_group": PATHOLOGY_CLASSES[patient_true[pid]],
            "predicted_group": PATHOLOGY_CLASSES[pred_idx],
            "correct": pred_idx == patient_true[pid],
            **{f"prob_{c}": mean_probs[i].item() for i, c in enumerate(PATHOLOGY_CLASSES)},
        })

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(output_dir, "test_patient_predictions.csv")
    results_df.to_csv(results_csv, index=False)

    overall_accuracy = results_df["correct"].mean()

    confusion = pd.crosstab(results_df["true_group"], results_df["predicted_group"]).reindex(
        index=PATHOLOGY_CLASSES, columns=PATHOLOGY_CLASSES, fill_value=0
    )
    confusion_csv = os.path.join(output_dir, "test_confusion_matrix.csv")
    confusion.to_csv(confusion_csv)

    per_class_rows = []
    for c in PATHOLOGY_CLASSES:
        tp = int(((results_df.true_group == c) & (results_df.predicted_group == c)).sum())
        fp = int(((results_df.true_group != c) & (results_df.predicted_group == c)).sum())
        fn = int(((results_df.true_group == c) & (results_df.predicted_group != c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        per_class_rows.append({"group": c, "precision": precision, "recall": recall, "f1": f1, "n_patients": tp + fn})
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = os.path.join(output_dir, "test_per_class_metrics.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    report_lines = [
        "# Learned MRI pathology classifier - end-to-end test validation",
        "",
        f"Evaluated on {len(results_df)} held-out test patients (never seen during YOLO/U-Net/classifier "
        "training or model selection). Predictions aggregated per patient (mean softmax probability across "
        "all of that patient's ED+ES slices, then argmax) - not a slice-level number.",
        "",
        f"**Overall per-patient accuracy: {overall_accuracy:.4f}** ({int(results_df['correct'].sum())}/{len(results_df)}), "
        f"chance level = {1/len(PATHOLOGY_CLASSES):.2f} for {len(PATHOLOGY_CLASSES)} classes.",
        "",
        "## Per-class precision/recall/F1",
        "",
        "| Group | Precision | Recall | F1 | N patients |",
        "|---|---|---|---|---|",
    ]
    for _, r in per_class_df.iterrows():
        report_lines.append(f"| {r['group']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['n_patients']} |")

    report_lines += ["", "## Confusion matrix (rows = true, columns = predicted)", "", confusion.to_markdown()]

    report_path = os.path.join(output_dir, "classifier_test_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "overall_accuracy": overall_accuracy,
        "per_class": per_class_df.to_dict("records"),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_test_validation(), indent=2, default=str))
