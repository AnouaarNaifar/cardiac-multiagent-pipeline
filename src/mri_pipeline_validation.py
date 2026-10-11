"""
Step 5 - end-to-end validation of the trained YOLO+U-Net pipeline
=====================================================================
The honest, citable check from results/report_package/05_future_work_raw_mri_pipeline/
raw_mri_pipeline_proposal.md: for every held-out TEST patient (50 patients,
never touched during YOLO/U-Net training or model selection), predicts heart
location (YOLO) and segmentation (U-Net) from the RAW MRI ALONE - no expert
mask used anywhere in this path, simulating the real deployment scenario -
then recomputes chamber volumes/ejection fractions/flags with mri_agent.py's
and mri_preprocessing.py's EXISTING, already-validated logic (reused
unchanged), and compares against the same pathology-group ground truth
mri_flag_validation.py already used.

Two features CSVs are produced, in the exact shape
mri_flag_validation.evaluate_mri_flags_against_pathology_groups expects, so
that module's flag-validation logic is reused unchanged (not duplicated):
  - expert-mask baseline, restricted to the same 50 test patients (a fair
    apples-to-apples comparison point - NOT the original all-150 number)
  - predicted-mask pipeline (the real, new result)
Also reports per-patient Dice (LV/RV/myocardium) between the predicted and
expert mask volumes, as a segmentation-only sanity check independent of the
downstream flags.
"""

import os

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from ultralytics import YOLO

from mri_flag_validation import evaluate_mri_flags_against_pathology_groups, write_mri_flag_validation_report
from mri_preprocessing import (
    compute_chamber_volumes_ml,
    compute_ejection_fraction_pct,
    load_nifti_volume,
    load_patient_info,
    normalize_slice_to_uint8,
)
from mri_yolo_dataset import build_train_val_test_split

_UNET_IMAGE_SIZE = 256
_YOLO_WEIGHTS = "results/mri_yolo_training/final/weights/best.pt"
_UNET_WEIGHTS = "results/mri_unet_training/best.pt"


def _load_models(device):
    yolo_model = YOLO(_YOLO_WEIGHTS)
    unet_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=4)
    unet_model.load_state_dict(torch.load(_UNET_WEIGHTS, map_location=device))
    unet_model.to(device).eval()
    return yolo_model, unet_model


def _predict_bbox(image_2d_uint8, yolo_model, conf_threshold=0.25):
    """Predict the heart bbox as (top, bottom, left, right); fall back to the
    full image extent if YOLO detects nothing above conf_threshold (mirrors
    compute_bounding_box's own no-foreground fallback behavior)."""
    rgb = np.stack([image_2d_uint8] * 3, axis=-1)
    results = yolo_model.predict(source=rgb, verbose=False, conf=conf_threshold)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return (0, image_2d_uint8.shape[0], 0, image_2d_uint8.shape[1])
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    height, width = image_2d_uint8.shape
    top = max(0, int(round(y1)))
    bottom = min(height, int(round(y2)))
    left = max(0, int(round(x1)))
    right = min(width, int(round(x2)))
    if bottom <= top or right <= left:
        return (0, height, 0, width)
    return (top, bottom, left, right)


def _predict_mask_slice(image_2d_uint8, yolo_model, unet_model, device):
    """Full inference chain for one raw slice: YOLO bbox -> crop -> U-Net
    segmentation -> paste back into a full-size predicted mask."""
    bbox = _predict_bbox(image_2d_uint8, yolo_model)
    top, bottom, left, right = bbox

    crop = image_2d_uint8[top:bottom, left:right]
    crop_pil = Image.fromarray(crop).resize((_UNET_IMAGE_SIZE, _UNET_IMAGE_SIZE), Image.BILINEAR)
    input_tensor = torch.from_numpy(np.array(crop_pil, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = unet_model(input_tensor)
        pred_crop = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_crop_resized = np.array(
        Image.fromarray(pred_crop).resize((right - left, bottom - top), Image.NEAREST)
    )

    full_mask = np.zeros_like(image_2d_uint8, dtype=np.uint8)
    full_mask[top:bottom, left:right] = pred_crop_resized
    return full_mask


def _predict_patient_phase_volume(patient_dir, patient_id, frame, yolo_model, unet_model, device):
    volume, voxel_spacing_mm = load_nifti_volume(os.path.join(patient_dir, f"{patient_id}_frame{frame:02d}.nii.gz"))
    n_slices = volume.shape[2]
    predicted_mask = np.zeros_like(volume, dtype=np.uint8)
    for z in range(n_slices):
        image_2d = normalize_slice_to_uint8(volume[:, :, z].T)
        pred_slice = _predict_mask_slice(image_2d, yolo_model, unet_model, device)
        predicted_mask[:, :, z] = pred_slice.T
    return predicted_mask, voxel_spacing_mm


def _dice_score(pred_mask, gt_mask, label):
    pred_c = (pred_mask == label)
    gt_c = (gt_mask == label)
    denom = pred_c.sum() + gt_c.sum()
    if denom == 0:
        return None
    return float(2 * np.logical_and(pred_c, gt_c).sum() / denom)


def run_end_to_end_validation(acdc_root="ACDC/database", output_dir="results/mri_pipeline_validation"):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model, unet_model = _load_models(device)

    test_patients = build_train_val_test_split(acdc_root)["test"]

    predicted_rows = []
    dice_rows = []
    for patient_id, patient_dir, group in test_patients:
        info = load_patient_info(os.path.join(patient_dir, "Info.cfg"))

        pred_ed_mask, voxel_spacing_mm = _predict_patient_phase_volume(
            patient_dir, patient_id, info["ED"], yolo_model, unet_model, device
        )
        pred_es_mask, _ = _predict_patient_phase_volume(
            patient_dir, patient_id, info["ES"], yolo_model, unet_model, device
        )

        gt_ed_mask, _ = load_nifti_volume(os.path.join(patient_dir, f"{patient_id}_frame{info['ED']:02d}_gt.nii.gz"))
        gt_es_mask, _ = load_nifti_volume(os.path.join(patient_dir, f"{patient_id}_frame{info['ES']:02d}_gt.nii.gz"))

        for phase, pred_mask, gt_mask in (("ed", pred_ed_mask, gt_ed_mask), ("es", pred_es_mask, gt_es_mask)):
            for label, name in ((1, "rv"), (2, "myocardium"), (3, "lv")):
                dice = _dice_score(pred_mask, gt_mask, label)
                dice_rows.append({"patient_id": patient_id, "group": group, "phase": phase, "structure": name, "dice": dice})

        ed_volumes = compute_chamber_volumes_ml(pred_ed_mask, voxel_spacing_mm)
        es_volumes = compute_chamber_volumes_ml(pred_es_mask, voxel_spacing_mm)
        lv_ef = compute_ejection_fraction_pct(ed_volumes["lv_volume_ml"], es_volumes["lv_volume_ml"])
        rv_ef = compute_ejection_fraction_pct(ed_volumes["rv_volume_ml"], es_volumes["rv_volume_ml"])

        predicted_rows.append({
            "patient_id": patient_id, "group": group,
            "lv_volume_ed_ml": ed_volumes["lv_volume_ml"], "lv_volume_es_ml": es_volumes["lv_volume_ml"],
            "rv_volume_ed_ml": ed_volumes["rv_volume_ml"], "rv_volume_es_ml": es_volumes["rv_volume_ml"],
            "lv_ejection_fraction_pct": lv_ef, "rv_ejection_fraction_pct": rv_ef,
        })
        print(f"done: {patient_id} ({group}) lv_ef={lv_ef}", flush=True)

    predicted_df = pd.DataFrame(predicted_rows)
    predicted_csv = os.path.join(output_dir, "predicted_mask_features_test_set.csv")
    predicted_df.to_csv(predicted_csv, index=False)

    dice_df = pd.DataFrame(dice_rows)
    dice_csv = os.path.join(output_dir, "segmentation_dice_test_set.csv")
    dice_df.to_csv(dice_csv, index=False)

    all_features = pd.read_csv("results/mri_features_all_patients.csv")
    test_patient_ids = {p[0] for p in test_patients}
    expert_baseline_df = all_features[all_features.patient_id.isin(test_patient_ids)]
    expert_baseline_csv = os.path.join(output_dir, "expert_mask_features_test_subset.csv")
    expert_baseline_df.to_csv(expert_baseline_csv, index=False)

    _, predicted_summary = evaluate_mri_flags_against_pathology_groups(features_csv=predicted_csv)
    _, expert_summary = evaluate_mri_flags_against_pathology_groups(features_csv=expert_baseline_csv)

    write_mri_flag_validation_report(predicted_summary, len(test_patients), output_dir)
    os.rename(
        os.path.join(output_dir, "mri_flag_validation_report.md"),
        os.path.join(output_dir, "predicted_pipeline_flag_validation_report.md"),
    )
    write_mri_flag_validation_report(expert_summary, len(test_patients), output_dir)
    os.rename(
        os.path.join(output_dir, "mri_flag_validation_report.md"),
        os.path.join(output_dir, "expert_mask_baseline_flag_validation_report_test_subset.md"),
    )

    mean_dice_by_structure = dice_df.groupby("structure")["dice"].mean().to_dict()

    return {
        "predicted_summary": predicted_summary,
        "expert_baseline_summary": expert_summary,
        "mean_dice_by_structure": mean_dice_by_structure,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import json
    result = run_end_to_end_validation()
    print(json.dumps(result, indent=2, default=str))
