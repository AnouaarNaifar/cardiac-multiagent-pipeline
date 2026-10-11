"""
Full U-Net (segmentation_models_pytorch, resnet18 encoder) fine-tuning run -
heart segmentation (4-class: background/RV/myocardium/LV) on ACDC, trained
on jitter-augmented crops of the shared patient-disjoint train split,
validated (unjittered, ground-truth box) on the shared val split.

Checkpointing/early-stopping convention mirrors ecg_embedding.py's
train_embedding_model: atomic "latest.pt"/"best.pt", resumable in spirit,
epoch CSV log.
"""
import csv
import os
import sys
import time

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from mri_unet_dataset import ACDCSegmentationDataset

CHECKPOINT_DIR = "results/mri_unet_training"
EPOCHS = 50
PATIENCE = 15


def compute_mean_dice(preds, masks, classes=(1, 2, 3)):
    scores = []
    for c in classes:
        pred_c = (preds == c).float()
        mask_c = (masks == c).float()
        intersection = (pred_c * mask_c).sum()
        denom = pred_c.sum() + mask_c.sum()
        if denom > 0:
            scores.append((2 * intersection / denom).item())
    return sum(scores) / len(scores) if scores else 0.0


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    df = pd.read_csv("results/mri_yolo_dataset/manifest.csv")
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]

    train_ds = ACDCSegmentationDataset(
        train_df, "results/mri_yolo_dataset/train/images", "results/mri_unet_dataset/train/masks",
        image_size=256, jitter=True,
    )
    val_ds = ACDCSegmentationDataset(
        val_df, "results/mri_yolo_dataset/val/images", "results/mri_unet_dataset/val/masks",
        image_size=256, jitter=False,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=1, classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_log_path = os.path.join(CHECKPOINT_DIR, "epoch_log.csv")
    with open(epoch_log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "epoch_time_sec"])

    best_dice = -1.0
    epochs_without_improvement = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_t0 = time.time()
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = torch.argmax(model(images), dim=1)
                dice_scores.append(compute_mean_dice(preds, masks))
        val_dice = sum(dice_scores) / len(dice_scores)
        epoch_time = time.time() - epoch_t0

        print(f"epoch {epoch}/{EPOCHS}: train_loss={train_loss:.4f} val_dice={val_dice:.4f} time={epoch_time:.1f}s")
        with open(epoch_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_dice, epoch_time])

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val_dice improvement for {PATIENCE} epochs)")
                break

    print(f"UNET_TRAIN_TOTAL_TIME_SEC {time.time() - t0:.1f}")
    print(f"UNET_BEST_VAL_DICE {best_dice:.4f}")
    print("UNET_TRAINING_DONE")


if __name__ == "__main__":
    main()
