"""
Full training run for the learned MRI pathology classifier (image + U-Net-
predicted mask -> 5-class pathology probability). See mri_classifier_dataset.py
for the input design and why it uses the PREDICTED mask, not the expert one.

Training is per-slice (same convention as YOLO/U-Net - every foreground
slice of a patient is a separate training example, inheriting that
patient's group label). Validation is aggregated PER PATIENT (mean
predicted probability across all of that patient's val slices, then
argmax) - matches the real deployment output ("this patient is likely
class X"), not a slice-level number that doesn't correspond to anything
clinically meaningful on its own.
"""
import csv
import os
import sys
import time

import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from mri_classifier_dataset import ACDCPathologyClassificationDataset, PATHOLOGY_CLASSES

CHECKPOINT_DIR = "results/mri_classifier_training"
EPOCHS = 50
PATIENCE = 15
N_INPUT_CHANNELS = 5  # 1 image + 4 one-hot mask classes


class PathologyClassifier(nn.Module):
    def __init__(self, n_classes=len(PATHOLOGY_CLASSES)):
        super().__init__()
        self.encoder = smp.encoders.get_encoder("resnet18", in_channels=N_INPUT_CHANNELS, weights="imagenet")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.encoder.out_channels[-1], n_classes)

    def forward(self, x):
        features = self.encoder(x)[-1]
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)


def per_patient_accuracy(model, loader, device):
    model.eval()
    patient_probs = {}
    patient_true = {}
    with torch.no_grad():
        for images, labels, patient_ids in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu()
            for i, pid in enumerate(patient_ids):
                patient_probs.setdefault(pid, []).append(probs[i])
                patient_true[pid] = labels[i].item()

    correct = 0
    for pid, probs_list in patient_probs.items():
        mean_probs = torch.stack(probs_list).mean(dim=0)
        pred = int(mean_probs.argmax())
        if pred == patient_true[pid]:
            correct += 1
    return correct / len(patient_probs)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    df = pd.read_csv("results/mri_yolo_dataset/manifest.csv")
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]

    train_ds = ACDCPathologyClassificationDataset(train_df, image_size=256, augment=True)
    val_ds = ACDCPathologyClassificationDataset(val_df, image_size=256, augment=False)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PathologyClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epoch_log_path = os.path.join(CHECKPOINT_DIR, "epoch_log.csv")
    with open(epoch_log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_patient_accuracy", "epoch_time_sec"])

    best_acc = -1.0
    epochs_without_improvement = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        model.train()
        total_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        val_acc = per_patient_accuracy(model, val_loader, device)
        epoch_time = time.time() - epoch_t0

        print(f"epoch {epoch}/{EPOCHS}: train_loss={train_loss:.4f} val_patient_acc={val_acc:.4f} time={epoch_time:.1f}s", flush=True)
        with open(epoch_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_acc, epoch_time])

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val_patient_acc improvement for {PATIENCE} epochs)")
                break

    print(f"CLASSIFIER_TRAIN_TOTAL_TIME_SEC {time.time() - t0:.1f}")
    print(f"CLASSIFIER_BEST_VAL_PATIENT_ACC {best_acc:.4f}")
    print("CLASSIFIER_TRAINING_DONE")


if __name__ == "__main__":
    main()
