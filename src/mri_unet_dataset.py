"""
U-Net segmentation dataset - masks aligned with the existing YOLO bbox dataset
=================================================================================
Step 2 data prep (see results/report_package/05_future_work_raw_mri_pipeline/
raw_mri_pipeline_proposal.md): builds the per-pixel segmentation targets the
U-Net stage needs, reusing the EXACT same images and patient-disjoint split
already extracted by mri_yolo_dataset.py (same file stems, same manifest,
same train/val/test patient assignment - no new split is computed here,
avoiding any risk of the two stages drifting apart).

extract_masks_for_split() re-reads each slice's ACDC ground-truth mask
(0=background, 1=RV, 2=myocardium, 3=LV - mri_preprocessing.py's label
convention) and caches it as a .npy next to the already-extracted PNG image,
under the same file stem - one .npy load per slice is far cheaper than
re-parsing the source .nii.gz volume on every training epoch.

ACDCSegmentationDataset applies box jitter (random scale + translation) to
the manifest's stored ground-truth bounding box **only for the train
split** - simulating YOLO's real imprecision at inference, so the segmenter
is never trained on a perfectly-cropped box it will never see in
production. val/test use the exact ground-truth box unjittered (a clean
baseline for model selection); the true end-to-end accuracy against a
REAL trained YOLO's predicted boxes is a separate, later validation step
(step 5 in the proposal doc), not this dataset's job.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from mri_preprocessing import load_nifti_volume

_YOLO_DATASET_DIR = "results/mri_yolo_dataset"
_UNET_DATASET_DIR = "results/mri_unet_dataset"
_ACDC_ROOT = "ACDC/database"


def _find_patient_dir(acdc_root, patient_id):
    for split_dir in ("training", "testing"):
        candidate = os.path.join(acdc_root, split_dir, patient_id)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate {patient_id} under {acdc_root}")


def extract_masks_for_split(acdc_root=_ACDC_ROOT, manifest_csv=None, output_dir=_UNET_DATASET_DIR):
    """
    Cache one .npy segmentation mask per slice already present in
    mri_yolo_dataset's manifest, under the same file stem, split into the
    same train/val/test directories.

    Output:
        dict: {"n_masks_written": int, "output_dir": str}.
    """
    manifest_csv = manifest_csv or os.path.join(_YOLO_DATASET_DIR, "manifest.csv")
    df = pd.read_csv(manifest_csv)

    n_written = 0
    # Group by (patient_id, phase) so each phase's NIfTI volume is loaded once,
    # not once per slice.
    for (patient_id, phase), group in df.groupby(["patient_id", "phase"]):
        patient_dir = _find_patient_dir(acdc_root, patient_id)
        info_path = os.path.join(patient_dir, "Info.cfg")
        from mri_preprocessing import load_patient_info
        info = load_patient_info(info_path)
        frame = info["ED"] if phase == "ed" else info["ES"]
        mask_vol, _ = load_nifti_volume(os.path.join(patient_dir, f"{patient_id}_frame{frame:02d}_gt.nii.gz"))

        for _, row in group.iterrows():
            mask_2d = mask_vol[:, :, row["slice_index"]].T.astype(np.uint8)
            split_dir = os.path.join(output_dir, row["split"], "masks")
            os.makedirs(split_dir, exist_ok=True)
            stem = os.path.splitext(row["image_filename"])[0]
            np.save(os.path.join(split_dir, f"{stem}.npy"), mask_2d)
            n_written += 1

    return {"n_masks_written": n_written, "output_dir": output_dir}


def _jitter_bbox(bbox, image_shape, scale_range=(0.85, 1.15), translate_frac=0.10, rng=random):
    top, bottom, left, right = bbox
    height, width = image_shape
    box_h, box_w = bottom - top, right - left
    cy, cx = (top + bottom) / 2.0, (left + right) / 2.0

    scale = rng.uniform(*scale_range)
    new_h, new_w = box_h * scale, box_w * scale
    cy += rng.uniform(-translate_frac, translate_frac) * box_h
    cx += rng.uniform(-translate_frac, translate_frac) * box_w

    new_top = max(0, int(round(cy - new_h / 2)))
    new_left = max(0, int(round(cx - new_w / 2)))
    new_bottom = min(height, int(round(cy + new_h / 2)))
    new_right = min(width, int(round(cx + new_w / 2)))
    if new_bottom <= new_top:
        new_bottom = min(height, new_top + 1)
    if new_right <= new_left:
        new_right = min(width, new_left + 1)
    return (new_top, new_bottom, new_left, new_right)


class ACDCSegmentationDataset(Dataset):
    """
    PyTorch Dataset over one split's already-extracted (image, mask) pairs,
    cropped to the (optionally jittered) heart bounding box and resized to
    a fixed training resolution.

    Inputs:
        manifest_df: rows for exactly one split (filter mri_yolo_dataset's
            manifest.csv by "split" before constructing this).
        images_dir / masks_dir: split-specific "images"/"masks" directories
            (mri_yolo_dataset's images/, mri_unet_dataset's masks/).
        image_size: output height/width (square) after crop+resize.
        jitter: if True, apply random scale/translation to the ground-truth
            box before cropping (train split only, by convention).
    """

    def __init__(self, manifest_df, images_dir, masks_dir, image_size=256, jitter=False, seed=0):
        self.rows = manifest_df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.jitter = jitter
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        stem = os.path.splitext(row["image_filename"])[0]

        image = np.array(Image.open(os.path.join(self.images_dir, row["image_filename"])))
        mask = np.load(os.path.join(self.masks_dir, f"{stem}.npy"))

        bbox = (row["bbox_top"], row["bbox_bottom"], row["bbox_left"], row["bbox_right"])
        if self.jitter:
            bbox = _jitter_bbox(bbox, image.shape, rng=self.rng)
        top, bottom, left, right = bbox

        image_crop = image[top:bottom, left:right]
        mask_crop = mask[top:bottom, left:right]

        image_pil = Image.fromarray(image_crop).resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray(mask_crop).resize((self.image_size, self.image_size), Image.NEAREST)

        image_tensor = torch.from_numpy(np.array(image_pil, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        return image_tensor, mask_tensor


if __name__ == "__main__":
    print(extract_masks_for_split())
