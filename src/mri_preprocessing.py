"""
MRI (ACDC) preprocessing pipeline
===================================
Mirrors ecg_preprocessing.py's role for the ECG modality: turns a raw ACDC
patient record into (a) a representative 2D image per cardiac phase, ready
to send to MedGemma's vision encoder, and (b) mask-derived numeric features
(chamber volumes, ejection fractions), the same way ecg_preprocessing.py
turns a raw signal into R-peaks/RR-intervals.

ACDC layout (per patient directory, e.g. ".../patient001"):
    Info.cfg                  - "ED: <frame>", "ES: <frame>", "Group: <label>", ...
    patientXXX_frameNN.nii.gz     - one cardiac phase, already extracted as a
                                     standalone 3D short-axis volume (X, Y, Z slices)
    patientXXX_frameNN_gt.nii.gz  - matching segmentation mask, labels:
                                     0 = background, 1 = RV cavity,
                                     2 = myocardium, 3 = LV cavity

Slice selection: rather than a fixed slice index (unsafe - the number of
slices and where the heart sits in the stack both vary per patient, and the
outermost slices often contain no heart tissue at all, confirmed on
patient001 where slice 0 has zero foreground voxels****************), the slice with the
most total foreground mask area is picked per phase - a simple, deterministic
choice that only requires an argmax over the (already available) ground
truth mask, not a learned or hand-tuned rule.

Dependencies beyond the rest of the project: nibabel (NIfTI), Pillow (image
encoding) - both already installed in this project's environment.
    pip install nibabel Pillow
"""

import base64
import io
import os

import nibabel as nib
import numpy as np
from PIL import Image

LABEL_BACKGROUND = 0
LABEL_RV = 1
LABEL_MYOCARDIUM = 2
LABEL_LV = 3


# ----------------------------------------------------------------------
# 1. Patient metadata (Info.cfg)
# ----------------------------------------------------------------------
def load_patient_info(info_cfg_path):
    """
    Parse an ACDC "Info.cfg" file into a dict.

    Inputs:
        info_cfg_path: path to a patient's Info.cfg file.

    Output:
        dict with keys "ED" and "ES" (ints, frame numbers), "Group" (str,
        e.g. "DCM"/"NOR"/"MINF"/"HCM"/"RV"), "Height", "Weight" (floats),
        "NbFrame" (int).
    """
    info = {}
    with open(info_cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if key in ("ED", "ES", "NbFrame"):
                info[key] = int(value)
            elif key in ("Height", "Weight"):
                info[key] = float(value)
            else:
                info[key] = value
    return info


# ----------------------------------------------------------------------
# 2. NIfTI loading
# ----------------------------------------------------------------------
def load_nifti_volume(path):
    """
    Load a NIfTI (.nii.gz) file as a 3D numpy array plus its voxel spacing.

    Inputs:
        path: path to a .nii.gz file.

    Output:
        (volume, voxel_spacing_mm) tuple: volume is a (X, Y, Z) float64 numpy
        array; voxel_spacing_mm is a 3-tuple of voxel dimensions in mm.
    """
    img = nib.load(path)
    volume = img.get_fdata()
    voxel_spacing_mm = img.header.get_zooms()[:3]
    return volume, voxel_spacing_mm


# ----------------------------------------------------------------------
# 3. Slice selection
# ----------------------------------------------------------------------
def select_representative_slice(mask_volume):
    """
    Pick the short-axis slice index with the most total foreground mask
    area (RV + myocardium + LV voxels combined), as a simple, deterministic
    proxy for "most representative of the heart" (typically a mid-ventricle
    slice) - avoids assuming a fixed slice index, which is unsafe since
    stack size and heart position both vary per patient.

    Inputs:
        mask_volume: (X, Y, Z) segmentation mask array (ACDC label
            convention: 0=background, 1=RV, 2=myocardium, 3=LV).

    Output:
        Integer slice index (along the Z axis).
    """
    foreground_area_per_slice = np.sum(mask_volume > LABEL_BACKGROUND, axis=(0, 1))
    return int(np.argmax(foreground_area_per_slice))


# ----------------------------------------------------------------------
# 4. ROI cropping (display images only - never used for volume computation)
# ----------------------------------------------------------------------
def compute_bounding_box(mask_2d, margin_fraction=0.15):
    """
    Compute a bounding box around all non-background pixels (RV +
    myocardium + LV combined) in a 2D segmentation mask slice, expanded by a
    margin proportional to the box's own size. Used to crop the display
    image around the heart so the vision model isn't shown mostly
    background when its encoder resizes the image internally - NOT used for
    any volume/feature computation, which always operates on the full mask.

    Inputs:
        mask_2d: 2D segmentation mask array (ACDC label convention:
            0=background, 1=RV, 2=myocardium, 3=LV).
        margin_fraction: fraction of the bounding box's larger dimension
            (max(height, width)) added as margin on each side. Proportional
            rather than a fixed pixel count, since slice resolution varies
            across patients/scanners.

    Output:
        (top, bottom, left, right) tuple of ints - row/column bounds such
        that mask_2d[top:bottom, left:right] contains the heart plus
        margin (bottom and right are exclusive, Python slice convention).
        Clamped to the mask's actual bounds. If the mask has zero
        foreground pixels, returns the full extent of mask_2d (i.e. no
        crop) instead of raising an exception.
    """
    rows, cols = np.where(mask_2d > LABEL_BACKGROUND)
    if rows.size == 0 or cols.size == 0:
        return (0, mask_2d.shape[0], 0, mask_2d.shape[1])

    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1

    box_height = bottom - top
    box_width = right - left
    margin = int(round(margin_fraction * max(box_height, box_width)))

    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(mask_2d.shape[0], bottom + margin)
    right = min(mask_2d.shape[1], right + margin)

    return (top, bottom, left, right)


def crop_to_bounding_box(slice_2d, bbox):
    """
    Crop a 2D array (image or mask) to a given bounding box.

    Inputs:
        slice_2d: 2D numpy array to crop.
        bbox: (top, bottom, left, right) tuple as returned by
            compute_bounding_box (bottom/right exclusive).

    Output:
        Cropped 2D numpy array (a view into slice_2d, not a copy).
    """
    top, bottom, left, right = bbox
    return slice_2d[top:bottom, left:right]


def union_bounding_boxes(bbox_a, bbox_b, max_rows=None, max_cols=None):
    """
    Combine two bounding boxes into the smallest box containing both. Used
    to give the ED and ES display images the same crop window (instead of
    each being cropped independently to its own tightest fit), so the two
    images share the same scale/framing and are directly comparable
    side-by-side - independent per-phase cropping would otherwise zoom each
    phase differently, making it harder to visually judge how much the
    ventricle changed between phases.

    Inputs:
        bbox_a: (top, bottom, left, right) tuple.
        bbox_b: (top, bottom, left, right) tuple.
        max_rows: optional cap on "bottom" (e.g. if bbox_a and bbox_b were
            computed on slices of slightly different shape, to keep the
            union valid for both). If None, no extra cap is applied beyond
            the two input boxes' own bounds.
        max_cols: optional cap on "right", analogous to max_rows.

    Output:
        (top, bottom, left, right) tuple: the union of both boxes, clamped
        to [0, max_rows) / [0, max_cols) if those are given.
    """
    top_a, bottom_a, left_a, right_a = bbox_a
    top_b, bottom_b, left_b, right_b = bbox_b

    top = max(0, min(top_a, top_b))
    left = max(0, min(left_a, left_b))
    bottom = max(bottom_a, bottom_b)
    right = max(right_a, right_b)

    if max_rows is not None:
        bottom = min(bottom, max_rows)
    if max_cols is not None:
        right = min(right, max_cols)

    return (top, bottom, left, right)


# ----------------------------------------------------------------------
# 5. Intensity normalization and image encoding
# ----------------------------------------------------------------------
def normalize_slice_to_uint8(slice_2d):
    """
    Min-max normalize a single 2D MRI slice to the 0-255 uint8 range
    expected by standard image formats (MRI intensities aren't naturally
    bounded like natural-image pixel values).

    Inputs:
        slice_2d: 2D numpy array (raw MRI intensities).

    Output:
        2D uint8 numpy array, same shape as the input.
    """
    slice_2d = slice_2d.astype(np.float64)
    min_val, max_val = slice_2d.min(), slice_2d.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    normalized = (slice_2d - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def encode_slice_to_base64_png(slice_2d_uint8):
    """
    Encode a 2D uint8 grayscale array as a PNG and return it base64-encoded,
    ready for Ollama's "images" field (see agent_common.generate_report_medgemma).

    Inputs:
        slice_2d_uint8: 2D uint8 numpy array.

    Output:
        Base64-encoded PNG string (str).
    """
    img = Image.fromarray(slice_2d_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_slice_png(slice_2d_uint8, output_path):
    """
    Save a 2D uint8 grayscale array as a PNG file on disk, for visual
    sanity-checking before ever sending it to a model.

    Inputs:
        slice_2d_uint8: 2D uint8 numpy array.
        output_path: file path to save the PNG to.

    Output:
        The output_path (str), for convenience.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(slice_2d_uint8, mode="L").save(output_path)
    return output_path


# ----------------------------------------------------------------------
# 6. Mask-derived numeric features (chamber volumes, ejection fractions)
# ----------------------------------------------------------------------
def compute_chamber_volumes_ml(mask_volume, voxel_spacing_mm):
    """
    Compute RV/myocardium/LV volumes (mL) from a full 3D segmentation mask,
    using ACDC's label convention. Uses the whole volume (all slices), not
    just the representative slice picked for display, since clinical
    ejection fraction is computed from the full ventricular volume.

    Inputs:
        mask_volume: (X, Y, Z) segmentation mask array for one cardiac phase.
        voxel_spacing_mm: 3-tuple of voxel dimensions in mm (from
            load_nifti_volume).

    Output:
        dict with "rv_volume_ml", "myocardium_volume_ml", "lv_volume_ml" (all
        native Python floats, not numpy scalars - needed for JSON
        serialization elsewhere, e.g. when a contribution's trace embeds
        this dict).
    """
    # voxel_spacing_mm entries come from nibabel as numpy.float32; under
    # NumPy's current scalar promotion rules, a Python float multiplied by a
    # numpy.float32 stays float32 rather than upcasting - so the final
    # float(...) wrap below isn't redundant, it's required.
    voxel_volume_mm3 = float(voxel_spacing_mm[0]) * float(voxel_spacing_mm[1]) * float(voxel_spacing_mm[2])
    return {
        "rv_volume_ml": float(float(np.sum(mask_volume == LABEL_RV)) * voxel_volume_mm3 / 1000.0),
        "myocardium_volume_ml": float(
            float(np.sum(mask_volume == LABEL_MYOCARDIUM)) * voxel_volume_mm3 / 1000.0
        ),
        "lv_volume_ml": float(float(np.sum(mask_volume == LABEL_LV)) * voxel_volume_mm3 / 1000.0),
    }


def compute_ejection_fraction_pct(volume_ed_ml, volume_es_ml):
    """
    Compute ejection fraction as a percentage: (ED volume - ES volume) / ED
    volume * 100. Standard definition, applicable to either the LV or RV
    cavity volume.

    Inputs:
        volume_ed_ml: cavity volume (mL) at end-diastole.
        volume_es_ml: cavity volume (mL) at end-systole.

    Output:
        Ejection fraction as a percentage (float), or None if volume_ed_ml
        is zero or negative (not computable).
    """
    if volume_ed_ml <= 0:
        return None
    return (volume_ed_ml - volume_es_ml) / volume_ed_ml * 100.0


# ----------------------------------------------------------------------
# 7. Full per-patient preprocessing pipeline
# ----------------------------------------------------------------------
def preprocess_patient(patient_dir):
    """
    Run the full ACDC preprocessing pipeline for one patient: load ED/ES
    frames and masks, pick a representative slice per phase, normalize and
    base64-encode it for the vision model, and compute mask-derived numeric
    features (chamber volumes and ejection fractions).

    Inputs:
        patient_dir: path to a patientXXX directory (e.g.
            "ACDC/database/training/patient001").

    Output:
        dict with:
            "patient_id": patient directory name (str).
            "info": dict from load_patient_info (ED/ES frame numbers, Group, ...).
            "images": {"short_axis_mid_ed": base64 PNG str,
                       "short_axis_mid_es": base64 PNG str} - each cropped to
                       the SAME ROI around the heart (the union of both
                       phases' individual bounding boxes - see
                       union_bounding_boxes) before normalization/encoding,
                       so ED and ES share the same scale/framing and are
                       directly comparable side-by-side; this crop is
                       display-only and never affects the numeric features
                       below.
            "slice_indices": {"ed": int, "es": int} - which slice was picked
                per phase, for traceability.
            "roi_bbox": {"ed": (top, bottom, left, right),
                         "es": (top, bottom, left, right)} - the shared
                bounding box used to crop both phases' display images (both
                entries are identical by construction), for traceability.
            "features": dict with rv/myocardium/lv volumes (mL) at ED and
                ES, plus "lv_ejection_fraction_pct" and
                "rv_ejection_fraction_pct" (None if not computable). Always
                computed from the full, uncropped 3D mask.
    """
    patient_id = os.path.basename(os.path.normpath(patient_dir))
    info = load_patient_info(os.path.join(patient_dir, "Info.cfg"))

    ed_frame = f"{info['ED']:02d}"
    es_frame = f"{info['ES']:02d}"

    ed_volume, voxel_spacing_mm = load_nifti_volume(
        os.path.join(patient_dir, f"{patient_id}_frame{ed_frame}.nii.gz")
    )
    ed_mask, _ = load_nifti_volume(
        os.path.join(patient_dir, f"{patient_id}_frame{ed_frame}_gt.nii.gz")
    )
    es_volume, _ = load_nifti_volume(
        os.path.join(patient_dir, f"{patient_id}_frame{es_frame}.nii.gz")
    )
    es_mask, _ = load_nifti_volume(
        os.path.join(patient_dir, f"{patient_id}_frame{es_frame}_gt.nii.gz")
    )

    ed_slice_idx = select_representative_slice(ed_mask)
    es_slice_idx = select_representative_slice(es_mask)

    # ROI cropping is display-only: computed from, and applied to, the same
    # 2D slice (mask and image, in the same transposed orientation) that
    # gets encoded for the vision model. compute_chamber_volumes_ml below
    # always receives the full, uncropped 3D mask - never affected by this.
    ed_mask_slice = ed_mask[:, :, ed_slice_idx].T
    es_mask_slice = es_mask[:, :, es_slice_idx].T

    # Each phase's own tight box is computed first, then unioned into one
    # shared box so ED and ES are cropped identically (same framing/scale)
    # rather than each being zoomed independently to its own tightest fit.
    ed_bbox_individual = compute_bounding_box(ed_mask_slice)
    es_bbox_individual = compute_bounding_box(es_mask_slice)
    shared_bbox = union_bounding_boxes(
        ed_bbox_individual,
        es_bbox_individual,
        max_rows=min(ed_mask_slice.shape[0], es_mask_slice.shape[0]),
        max_cols=min(ed_mask_slice.shape[1], es_mask_slice.shape[1]),
    )
    ed_bbox = es_bbox = shared_bbox

    ed_image_b64 = encode_slice_to_base64_png(
        normalize_slice_to_uint8(
            crop_to_bounding_box(ed_volume[:, :, ed_slice_idx].T, ed_bbox)
        )
    )
    es_image_b64 = encode_slice_to_base64_png(
        normalize_slice_to_uint8(
            crop_to_bounding_box(es_volume[:, :, es_slice_idx].T, es_bbox)
        )
    )

    ed_chamber_volumes = compute_chamber_volumes_ml(ed_mask, voxel_spacing_mm)
    es_chamber_volumes = compute_chamber_volumes_ml(es_mask, voxel_spacing_mm)

    lv_ef = compute_ejection_fraction_pct(
        ed_chamber_volumes["lv_volume_ml"], es_chamber_volumes["lv_volume_ml"]
    )
    rv_ef = compute_ejection_fraction_pct(
        ed_chamber_volumes["rv_volume_ml"], es_chamber_volumes["rv_volume_ml"]
    )

    return {
        "patient_id": patient_id,
        "info": info,
        "images": {
            "short_axis_mid_ed": ed_image_b64,
            "short_axis_mid_es": es_image_b64,
        },
        "slice_indices": {"ed": ed_slice_idx, "es": es_slice_idx},
        "roi_bbox": {"ed": ed_bbox, "es": es_bbox},
        "features": {
            "lv_volume_ed_ml": ed_chamber_volumes["lv_volume_ml"],
            "lv_volume_es_ml": es_chamber_volumes["lv_volume_ml"],
            "rv_volume_ed_ml": ed_chamber_volumes["rv_volume_ml"],
            "rv_volume_es_ml": es_chamber_volumes["rv_volume_ml"],
            "myocardium_volume_ed_ml": ed_chamber_volumes["myocardium_volume_ml"],
            "myocardium_volume_es_ml": es_chamber_volumes["myocardium_volume_ml"],
            "lv_ejection_fraction_pct": lv_ef,
            "rv_ejection_fraction_pct": rv_ef,
        },
    }
