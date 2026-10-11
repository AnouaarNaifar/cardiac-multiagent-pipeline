"""
Learned MRI pathology classifier - dataset (input = image + U-Net-predicted mask)
=====================================================================================
This is the missing piece discussed in the raw-MRI pipeline proposal (see
results/report_package/05_future_work_raw_mri_pipeline/raw_mri_pipeline_proposal.md):
YOLO+U-Net only localize/segment - nothing so far learns to recognize
pathology directly from the image. This module builds that dataset.

Input to the classifier = image (1 channel) + one-hot U-Net-PREDICTED mask
(4 channels: background/RV/myocardium/LV) = 5 channels total. Deliberately
the PREDICTED mask, not the ACDC expert mask: at real deployment the
classifier will only ever see U-Net's prediction, never the expert mask, so
training on the expert mask would be training on cleaner data than the
model will ever see in production - the same reasoning already applied to
U-Net's own jitter-augmented training.

cache_predicted_masks() runs the already-trained YOLO+U-Net chain (reused
from mri_pipeline_validation.py) ONCE per slice and caches the result -
avoids a full segmentation forward pass on every training epoch.

ACDCPathologyClassificationDataset then applies the agreed augmentation
recipe (rotation +-15, translation, zoom, brightness/contrast, light noise;
no flips - the heart is not symmetric) identically to the image and the
cached predicted mask together, on the fly, train split only.
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from mri_pipeline_validation import _load_models, _predict_mask_slice

_YOLO_DATASET_DIR = "results/mri_yolo_dataset"
_CLASSIFIER_DATASET_DIR = "results/mri_classifier_dataset"

# Fixed, deterministic class order - used everywhere a class index is needed.
PATHOLOGY_CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
_N_MASK_CLASSES = 4  # background, RV, myocardium, LV


def cache_predicted_masks(splits=("train", "val"), manifest_csv=None, output_dir=_CLASSIFIER_DATASET_DIR):
    """
    Run the trained YOLO+U-Net inference chain once per slice (for the given
    splits) and cache the predicted mask as a .npy, under the same file stem
    as the already-extracted image.

    Output:
        dict: {"n_masks_written": int, "output_dir": str}.
    """
    manifest_csv = manifest_csv or os.path.join(_YOLO_DATASET_DIR, "manifest.csv")
    df = pd.read_csv(manifest_csv)
    df = df[df.split.isin(splits)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model, unet_model = _load_models(device)

    n_written = 0
    for _, row in df.iterrows():
        image = np.array(Image.open(os.path.join(_YOLO_DATASET_DIR, row["split"], "images", row["image_filename"])))
        predicted_mask = _predict_mask_slice(image, yolo_model, unet_model, device)

        split_dir = os.path.join(output_dir, row["split"], "predicted_masks")
        os.makedirs(split_dir, exist_ok=True)
        stem = os.path.splitext(row["image_filename"])[0]
        np.save(os.path.join(split_dir, f"{stem}.npy"), predicted_mask)
        n_written += 1

    return {"n_masks_written": n_written, "output_dir": output_dir}


class ACDCPathologyClassificationDataset(Dataset):
    """
    PyTorch Dataset over one split's (image, predicted mask) pairs, one-hot
    encoding the mask and stacking it with the image into a 5-channel input,
    with the agreed augmentation recipe applied identically to both when
    jitter=True (train split only, by convention).
    """

    def __init__(self, manifest_df, yolo_dataset_dir=_YOLO_DATASET_DIR, classifier_dataset_dir=_CLASSIFIER_DATASET_DIR,
                 image_size=256, augment=False, seed=0):
        self.rows = manifest_df.reset_index(drop=True)
        self.yolo_dataset_dir = yolo_dataset_dir
        self.classifier_dataset_dir = classifier_dataset_dir
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.rows)

    def _augment_pair(self, image_pil, mask_pil):
        angle = self.rng.uniform(-15, 15)
        translate_frac = self.rng.uniform(-0.10, 0.10, size=2)
        scale = self.rng.uniform(0.90, 1.10)
        max_dx = translate_frac[0] * self.image_size
        max_dy = translate_frac[1] * self.image_size

        image_pil = TF.affine(image_pil, angle=angle, translate=(max_dx, max_dy), scale=scale, shear=0,
                               interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil = TF.affine(mask_pil, angle=angle, translate=(max_dx, max_dy), scale=scale, shear=0,
                              interpolation=TF.InterpolationMode.NEAREST)

        if self.rng.rand() < 0.8:
            brightness = self.rng.uniform(0.8, 1.2)
            contrast = self.rng.uniform(0.8, 1.2)
            image_pil = TF.adjust_brightness(image_pil, brightness)
            image_pil = TF.adjust_contrast(image_pil, contrast)

        return image_pil, mask_pil

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        stem = os.path.splitext(row["image_filename"])[0]

        image = np.array(Image.open(os.path.join(self.yolo_dataset_dir, row["split"], "images", row["image_filename"])))
        mask = np.load(os.path.join(self.classifier_dataset_dir, row["split"], "predicted_masks", f"{stem}.npy"))

        image_pil = Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            image_pil, mask_pil = self._augment_pair(image_pil, mask_pil)
            if self.rng.rand() < 0.5:
                noise = self.rng.normal(0, 6, size=(self.image_size, self.image_size))
                noisy = np.clip(np.array(image_pil, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
                image_pil = Image.fromarray(noisy)

        image_arr = np.array(image_pil, dtype=np.float32) / 255.0
        mask_arr = np.array(mask_pil, dtype=np.int64)
        mask_arr = np.clip(mask_arr, 0, _N_MASK_CLASSES - 1)
        mask_onehot = np.eye(_N_MASK_CLASSES, dtype=np.float32)[mask_arr].transpose(2, 0, 1)

        input_tensor = torch.from_numpy(np.concatenate([image_arr[None, :, :], mask_onehot], axis=0))
        label = PATHOLOGY_CLASSES.index(row["group"])
        return input_tensor, label, row["patient_id"]


def predict_pathology_probabilities(image_2d_uint8, yolo_model, unet_model, classifier_model, device="cpu", image_size=256):
    """
    Full inference chain for one raw MRI slice -> per-pathology-class
    probabilities: YOLO bbox -> crop -> U-Net predicted mask -> classifier.
    Mirrors ecg_embedding.predict_diagnostic_probabilities's role for the
    MRI side - the low-level inference entry point mri_agent.py's
    compute_pathology_classifier_features wraps.

    Inputs:
        image_2d_uint8: (H, W) uint8 array - the full, uncropped MRI slice.
        yolo_model / unet_model / classifier_model: loaded models (see
            mri_agent.compute_pathology_classifier_features).
        device: torch device string.
        image_size: resize target before classification (must match what
            the classifier was trained on - 256 by default).

    Output:
        dict of "pathology_<CLASS>_probability" -> float, one entry per
        PATHOLOGY_CLASSES.
    """
    predicted_mask = _predict_mask_slice(image_2d_uint8, yolo_model, unet_model, device)

    image_pil = Image.fromarray(image_2d_uint8).resize((image_size, image_size), Image.BILINEAR)
    mask_pil = Image.fromarray(predicted_mask).resize((image_size, image_size), Image.NEAREST)

    image_arr = np.array(image_pil, dtype=np.float32) / 255.0
    mask_arr = np.clip(np.array(mask_pil, dtype=np.int64), 0, _N_MASK_CLASSES - 1)
    mask_onehot = np.eye(_N_MASK_CLASSES, dtype=np.float32)[mask_arr].transpose(2, 0, 1)
    input_tensor = torch.from_numpy(
        np.concatenate([image_arr[None, :, :], mask_onehot], axis=0)
    ).unsqueeze(0).to(device)

    classifier_model.eval()
    with torch.no_grad():
        probs = torch.softmax(classifier_model(input_tensor), dim=1).squeeze(0).cpu().numpy()

    return {f"pathology_{cls}_probability": float(p) for cls, p in zip(PATHOLOGY_CLASSES, probs)}


if __name__ == "__main__":
    print(cache_predicted_masks())
