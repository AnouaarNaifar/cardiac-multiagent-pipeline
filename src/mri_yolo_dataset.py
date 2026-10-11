"""
YOLO bounding-box dataset extraction from ACDC's existing expert masks
=========================================================================
Step 1 of the planned raw-MRI pipeline extension (see
results/report_package/05_future_work_raw_mri_pipeline/raw_mri_pipeline_proposal.md):
produces the heart-localization training data a YOLO detector will need,
entirely from ACDC's already-provided expert segmentation masks - no manual
annotation, and no existing module modified (mri_preprocessing.py's
functions are reused unchanged, only imported).

For every ED/ES 2D slice (across all patients) that contains any heart
foreground (RV + myocardium + LV combined - the same single-box definition
mri_preprocessing.compute_bounding_box already uses for display cropping),
this writes:
  - a normalized-to-uint8 PNG of the raw MRI slice (the YOLO input image)
  - a matching YOLO-format label .txt (single class "heart": one
    "0 x_center y_center width height" line, all normalized to [0,1])
Slices with zero foreground are skipped (compute_bounding_box returns the
FULL slice extent when there is no foreground, which is not a real box and
would corrupt training if included as a label).

Split (fixed, reused by both YOLO and the later U-Net stage - do not
recompute independently for each model):
  - test  = ACDC's own official "testing" directory (50 patients, 10/group)
            - held out entirely, used ONLY for the final end-to-end
            validation (Dice + EF/flag precision-recall-F1 vs. pathology
            groups), never seen during training or model selection.
  - train/val = ACDC's own official "training" directory (100 patients,
            20/group), split 80/20 stratified per group (16/4 per group)
            with a fixed random seed - deterministic and reproducible.
This reuses ACDC's own official train/test directory split for the
outermost (train+val vs. test) boundary rather than inventing one, which is
both more defensible in the report and already exactly patient-disjoint and
group-balanced by construction (verified directly: 20/group in training/,
10/group in testing/).
"""

import csv
import os
import random
from pathlib import Path

import numpy as np

from mri_preprocessing import (
    compute_bounding_box,
    load_nifti_volume,
    load_patient_info,
    normalize_slice_to_uint8,
    save_slice_png,
)

_CLASS_NAME = "heart"


def _list_patients(split_dir):
    """Return sorted list of (patient_id, patient_dir, group) for one ACDC split directory."""
    patients = []
    for entry in sorted(os.listdir(split_dir)):
        patient_dir = os.path.join(split_dir, entry)
        info_path = os.path.join(patient_dir, "Info.cfg")
        if not entry.startswith("patient") or not os.path.isfile(info_path):
            continue
        info = load_patient_info(info_path)
        patients.append((entry, patient_dir, info["Group"]))
    return patients


def build_train_val_test_split(acdc_root, val_fraction=0.2, seed=42):
    """
    Build the fixed, shared patient-disjoint train/val/test split.

    Inputs:
        acdc_root: path to "ACDC/database" (contains "training"/"testing").
        val_fraction: fraction of the "training" directory's patients (per
            group) held out for validation; the remainder is train.
        seed: fixed random seed for the stratified per-group shuffle -
            deterministic and reproducible given the same inputs.

    Output:
        dict with keys "train", "val", "test", each a list of
        (patient_id, patient_dir, group) tuples.
    """
    train_pool = _list_patients(os.path.join(acdc_root, "training"))
    test = _list_patients(os.path.join(acdc_root, "testing"))

    by_group = {}
    for entry in train_pool:
        by_group.setdefault(entry[2], []).append(entry)

    rng = random.Random(seed)
    train, val = [], []
    for group in sorted(by_group):
        group_patients = sorted(by_group[group])
        rng.shuffle(group_patients)
        n_val = max(1, round(len(group_patients) * val_fraction))
        val.extend(group_patients[:n_val])
        train.extend(group_patients[n_val:])

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def _yolo_label_line(bbox, image_shape):
    top, bottom, left, right = bbox
    height, width = image_shape
    x_center = (left + right) / 2.0 / width
    y_center = (top + bottom) / 2.0 / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def _extract_patient_phase(patient_id, patient_dir, phase_frame, phase_label, images_dir, labels_dir, manifest_rows, group, split):
    volume, _ = load_nifti_volume(os.path.join(patient_dir, f"{patient_id}_frame{phase_frame}.nii.gz"))
    mask, _ = load_nifti_volume(os.path.join(patient_dir, f"{patient_id}_frame{phase_frame}_gt.nii.gz"))

    n_slices = volume.shape[2]
    for z in range(n_slices):
        mask_2d = mask[:, :, z].T
        if not np.any(mask_2d > 0):
            continue

        bbox = compute_bounding_box(mask_2d, margin_fraction=0.0)
        image_2d = normalize_slice_to_uint8(volume[:, :, z].T)

        file_stem = f"{split}_{patient_id}_{phase_label}_{z:02d}"
        save_slice_png(image_2d, os.path.join(images_dir, f"{file_stem}.png"))
        with open(os.path.join(labels_dir, f"{file_stem}.txt"), "w", encoding="utf-8") as f:
            f.write(_yolo_label_line(bbox, image_2d.shape) + "\n")

        manifest_rows.append({
            "image_filename": f"{file_stem}.png",
            "patient_id": patient_id,
            "group": group,
            "split": split,
            "phase": phase_label,
            "slice_index": z,
            "bbox_top": bbox[0], "bbox_bottom": bbox[1],
            "bbox_left": bbox[2], "bbox_right": bbox[3],
            "image_height": image_2d.shape[0], "image_width": image_2d.shape[1],
        })


def build_yolo_dataset(acdc_root="ACDC/database", output_dir="results/mri_yolo_dataset", val_fraction=0.2, seed=42):
    """
    Extract the full YOLO heart-localization dataset from ACDC's expert
    masks: images/labels for every foreground ED/ES slice, split into
    train/val/test per build_train_val_test_split, plus a manifest CSV and
    a ready-to-use Ultralytics "data.yaml".

    Output:
        dict: {"split_patient_counts": {...}, "split_image_counts": {...},
        "manifest_csv": path, "data_yaml": path}.
    """
    split_patients = build_train_val_test_split(acdc_root, val_fraction=val_fraction, seed=seed)

    output_dir = Path(output_dir)
    manifest_rows = []
    split_image_counts = {}

    for split, patients in split_patients.items():
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        before = len(manifest_rows)
        for patient_id, patient_dir, group in patients:
            info = load_patient_info(os.path.join(patient_dir, "Info.cfg"))
            _extract_patient_phase(
                patient_id, patient_dir, f"{info['ED']:02d}", "ed",
                images_dir, labels_dir, manifest_rows, group, split,
            )
            _extract_patient_phase(
                patient_id, patient_dir, f"{info['ES']:02d}", "es",
                images_dir, labels_dir, manifest_rows, group, split,
            )
        split_image_counts[split] = len(manifest_rows) - before

    manifest_csv = output_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(
            f"train: {(output_dir / 'train' / 'images').resolve()}\n"
            f"val: {(output_dir / 'val' / 'images').resolve()}\n"
            f"test: {(output_dir / 'test' / 'images').resolve()}\n"
            f"nc: 1\n"
            f"names: ['{_CLASS_NAME}']\n"
        )

    split_patient_counts = {split: len(patients) for split, patients in split_patients.items()}
    return {
        "split_patient_counts": split_patient_counts,
        "split_image_counts": split_image_counts,
        "manifest_csv": str(manifest_csv),
        "data_yaml": str(data_yaml),
    }


if __name__ == "__main__":
    result = build_yolo_dataset()
    print(result)
