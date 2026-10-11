"""
Pure helpers for the MRI page's raw-pipeline path (no Streamlit import).

Wraps the already-validated pieces of `mri_pipeline_validation.py` /
`mri_preprocessing.py` so the UI can, from a raw NIfTI volume alone:
  - predict a 3-D segmentation mask with the trained YOLO+U-Net (no expert mask),
  - derive chamber volumes + ejection fraction from that predicted mask,
  - (when an expert mask is also available) compare predicted vs expert EF and
    Dice - the "is our pipeline actually useful?" check the user asked for.

Because the predicted mask yields an EF, an uploaded MRI now has a computable
`lv_ejection_fraction_pct`, so `mri_agent.generate_mri_contribution` no longer
short-circuits on missing EF - the predicted features feed it exactly like an
ACDC expert-mask `preprocess_patient` result would.
"""

import numpy as np


def slice_axis_of(shape, voxel_spacing_mm):
    """Identify the through-plane (short-axis stack) axis of a 3-D cardiac
    volume. The rest of the pipeline assumes it is axis 2 (`volume[:, :, z]`),
    which is true for ACDC frames but NOT for an arbitrary upload. The robust
    cue is voxel spacing: the through-plane spacing is much larger than the
    in-plane one (e.g. 10 mm vs 1.5 mm). Falls back to the fewest-slices axis
    when spacing is isotropic/unavailable."""
    sp = [float(s) for s in voxel_spacing_mm[:3]]
    if len(sp) == 3 and min(sp) > 0 and max(sp) > 1.5 * min(sp):
        return int(np.argmax(sp))
    return int(np.argmin(shape[:3]))


def to_canonical(volume, voxel_spacing_mm):
    """Reorient a 3-D volume so the slice axis is last -> (X, Y, Z_slices), and
    reorder the spacing to match. Raises on 4-D input (a cine uploaded instead of
    a single ED/ES phase). Returns (volume_xyz, spacing_xyz, slice_axis, changed)."""
    if volume.ndim == 4:
        raise ValueError(
            f"Got a 4-D volume {volume.shape} - this looks like a 4-D cine "
            "(e.g. *_4d.nii.gz). Upload a single ED or ES phase frame, not the cine.")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {volume.shape}.")
    axis = slice_axis_of(volume.shape, voxel_spacing_mm)
    if axis == 2:
        return volume, tuple(float(s) for s in voxel_spacing_mm[:3]), axis, False
    vol = np.moveaxis(volume, axis, 2)
    order = [i for i in range(3) if i != axis] + [axis]  # matches np.moveaxis
    sp = tuple(float(voxel_spacing_mm[i]) for i in order)
    return vol, sp, axis, True


def move_slice_to_last(arr, slice_axis):
    """Apply the same reorientation as to_canonical to a companion array (e.g. an
    expert mask that shares the volume's geometry)."""
    return arr if slice_axis == 2 else np.moveaxis(arr, slice_axis, 2)


def predict_mask_volume(volume, yolo_model, unet_model, device="cpu"):
    """Run YOLO+U-Net slice-by-slice over a 3-D volume -> predicted mask volume,
    same convention as mri_pipeline_validation._predict_patient_phase_volume."""
    from mri_preprocessing import normalize_slice_to_uint8
    import mri_pipeline_validation as mpv
    pred = np.zeros_like(volume, dtype=np.uint8)
    for z in range(volume.shape[2]):
        img = normalize_slice_to_uint8(volume[:, :, z].T)
        pred[:, :, z] = mpv._predict_mask_slice(img, yolo_model, unet_model, device).T
    return pred


def features_from_masks(ed_mask, es_mask, voxel_spacing_mm):
    """Chamber volumes + EF in the SAME feature keys preprocess_patient produces,
    so the result can drop straight into mri_agent (works for a predicted OR an
    expert mask pair)."""
    from mri_preprocessing import compute_chamber_volumes_ml, compute_ejection_fraction_pct
    ed = compute_chamber_volumes_ml(ed_mask, voxel_spacing_mm)
    es = compute_chamber_volumes_ml(es_mask, voxel_spacing_mm)
    return {
        "lv_volume_ed_ml": ed["lv_volume_ml"], "lv_volume_es_ml": es["lv_volume_ml"],
        "rv_volume_ed_ml": ed["rv_volume_ml"], "rv_volume_es_ml": es["rv_volume_ml"],
        "myocardium_volume_ed_ml": ed["myocardium_volume_ml"],
        "myocardium_volume_es_ml": es["myocardium_volume_ml"],
        "lv_ejection_fraction_pct": compute_ejection_fraction_pct(
            ed["lv_volume_ml"], es["lv_volume_ml"]),
        "rv_ejection_fraction_pct": compute_ejection_fraction_pct(
            ed["rv_volume_ml"], es["rv_volume_ml"]),
    }


def dice_all(pred_mask, gt_mask):
    """Per-structure Dice (predicted vs expert mask) for the same ED phase."""
    import mri_pipeline_validation as mpv
    from mri_preprocessing import LABEL_RV, LABEL_MYOCARDIUM, LABEL_LV
    return {"RV": mpv._dice_score(pred_mask, gt_mask, LABEL_RV),
            "Myocardium": mpv._dice_score(pred_mask, gt_mask, LABEL_MYOCARDIUM),
            "LV": mpv._dice_score(pred_mask, gt_mask, LABEL_LV)}


def overlay_mask_on_slice(raw_uint8, mask_slice, alpha=0.45):
    """RGB overlay of the ACDC segmentation (1=RV, 2=myocardium, 3=LV) on a
    grayscale slice, so the user can visually judge the predicted mask. Returns
    an (H, W, 3) uint8 array. `mask_slice` must be in the SAME orientation as
    `raw_uint8` (i.e. pass pred_ed[:, :, idx].T alongside the .T-ed raw slice)."""
    base = np.stack([raw_uint8] * 3, axis=-1).astype(np.float32)
    over = base.copy()
    colors = {1: (70, 130, 255), 2: (70, 220, 130), 3: (255, 80, 80)}  # RV blue, Myo green, LV red
    for lab, col in colors.items():
        m = mask_slice == lab
        for c in range(3):
            over[..., c][m] = (1 - alpha) * base[..., c][m] + alpha * col[c]
    return over.clip(0, 255).astype(np.uint8)


def representative_ed(ed_volume, ed_mask):
    """Pick the ED slice with the most foreground in `ed_mask`, and return
    (slice_index, raw_uint8_slice, base64_png) for the image / classifier."""
    from mri_preprocessing import (select_representative_slice,
                                    normalize_slice_to_uint8,
                                    encode_slice_to_base64_png)
    idx = select_representative_slice(ed_mask)
    raw_uint8 = normalize_slice_to_uint8(ed_volume[:, :, idx].T)
    # Encode the NORMALIZED uint8 slice - encode_slice_to_base64_png does
    # Image.fromarray(..., mode="L") which needs uint8; handing it the raw float64
    # slice makes PIL reinterpret each 8-byte float as 8 grayscale pixels, which
    # renders as evenly-spaced vertical stripes (the display bug seen in the UI).
    return idx, raw_uint8, encode_slice_to_base64_png(raw_uint8)
