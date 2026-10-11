"""
ADAPTIVE preprocessing pipeline for ECG images, before sending to a VLM.

Principle: instead of binary rules (accept/reject), each correction is
applied PROPORTIONALLY to the severity of the detected issue. A clean
image passes through the pipeline almost unchanged; a degraded image
receives stronger corrections. A quality report (before/after) is
generated and meant to accompany the image in the VLM prompt, so the
model can adjust its confidence according to the actual reliability of
the source.

Cost: OpenCV/PIL/numpy only, no heavy model (deep learning) -
intentionally low-cost to stay fast and cheap to run.
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import base64
import io
import os

# ============================================================
# REFERENCE THRESHOLDS (adjustable based on real-world observations)
# ============================================================
TARGET_MAX_WIDTH = 1800          # target final width before sending to the VLM
MIN_ACCEPTABLE_WIDTH = 400        # below this: forced upscale
BLUR_SEVERE_THRESHOLD = 50.0      # Laplacian variance < threshold = severe blur
BLUR_MILD_THRESHOLD = 150.0       # between the two = mild blur
CONTRAST_LOW_THRESHOLD = 25.0     # pixel standard deviation
BRIGHTNESS_LOW = 60
BRIGHTNESS_HIGH = 200


# ============================================================
# STEP 1 — Loading and validation
# ============================================================
def load_and_validate(image_path):
    """Loads the image and checks that it is usable."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        pil_img = Image.open(image_path)
        pil_img.verify()
        pil_img = Image.open(image_path)  # reopen after verify()
    except Exception as e:
        raise ValueError(f"Unreadable/corrupted image: {e}")

    return pil_img


# ============================================================
# STEP 2 — Orientation correction (EXIF, for photos)
# ============================================================
def correct_orientation(pil_img):
    """Applies EXIF rotation if present (typical smartphone photo case)."""
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img, 0

        for tag_id, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag_id)
            if tag_name == "Orientation":
                rotation_map = {3: 180, 6: 270, 8: 90}
                angle = rotation_map.get(value, 0)
                if angle != 0:
                    pil_img = pil_img.rotate(-angle, expand=True)
                return pil_img, angle
        return pil_img, 0
    except (AttributeError, KeyError):
        return pil_img, 0  # no EXIF metadata (typically synthetic images)


# ============================================================
# STEP 3 — Flatten transparency (RGBA -> RGB)
# ============================================================
def flatten_to_rgb(pil_img):
    """Converts to pure RGB, flattening alpha onto a white background if
    present. Needed because some VLM APIs expect 3-channel RGB."""
    if pil_img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        rgba_img = pil_img.convert("RGBA")
        background.paste(rgba_img, mask=rgba_img.split()[-1])
        return background, True
    elif pil_img.mode != "RGB":
        return pil_img.convert("RGB"), True
    return pil_img, False


# ============================================================
# STEP 4 — Mild skew correction (best-effort, not full perspective warp)
# ============================================================
def deskew_if_needed(cv_img):
    """
    Detects and corrects a slight rotation (skew) via the largest contour.
    KNOWN LIMITATION: does NOT handle full perspective correction
    (trapezoid distortion from a photo taken at an angle) - only a simple
    rotation. See limitations section at the bottom of this file.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:  # not enough detected content, leave it alone
        return cv_img, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct minor skews (< 15°) - beyond that, risk of
    # over-correcting based on irrelevant content (text, borders...)
    if abs(angle) < 0.5 or abs(angle) > 15:
        return cv_img, 0.0

    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, round(float(angle), 2)


# ============================================================
# STEP 5 — Quality measurements (diagnostic)
# ============================================================
def measure_quality(cv_img):
    """Computes the quality metrics used to drive the corrections."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    height, width = gray.shape[:2]

    return {
        "width": width,
        "height": height,
        "blur_score": round(float(laplacian_var), 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
    }


# ============================================================
# STEP 6 — ADAPTIVE sharpness correction (proportional unsharp masking)
# ============================================================
def adaptive_sharpen(cv_img, blur_score):
    """
    Applies deblurring whose intensity depends on the measured severity.
    - Severe blur (< BLUR_SEVERE_THRESHOLD)  -> strong correction
    - Mild blur (< BLUR_MILD_THRESHOLD)      -> light correction
    - Sharp image                            -> no modification
    """
    if blur_score >= BLUR_MILD_THRESHOLD:
        return cv_img, 0.0  # already sharp, leave untouched

    if blur_score < BLUR_SEVERE_THRESHOLD:
        amount = 1.5   # strong correction
    else:
        amount = 0.7   # light correction

    gaussian = cv2.GaussianBlur(cv_img, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(cv_img, 1 + amount, gaussian, -amount, 0)
    return sharpened, amount


# ============================================================
# STEP 7 — ADAPTIVE contrast correction (proportional CLAHE)
# ============================================================
def adaptive_contrast(cv_img, contrast_score):
    """Applies CLAHE (local adaptive contrast) only when needed, with an
    intensity (clipLimit) proportional to the measured deficit."""
    if contrast_score >= CONTRAST_LOW_THRESHOLD:
        return cv_img, 0.0

    deficit_ratio = 1 - (contrast_score / CONTRAST_LOW_THRESHOLD)
    clip_limit = 1.0 + (deficit_ratio * 3.0)  # between 1.0 and 4.0

    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l)
    lab_corrected = cv2.merge((l_corrected, a, b))
    result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    return result, round(float(clip_limit), 2)


# ============================================================
# STEP 8 — ADAPTIVE brightness correction (proportional gamma)
# ============================================================
def adaptive_brightness(cv_img, brightness_score):
    """Corrects brightness via a gamma curve, only when needed."""
    if BRIGHTNESS_LOW <= brightness_score <= BRIGHTNESS_HIGH:
        return cv_img, 1.0

    target = 128.0
    gamma = np.log(target / 255.0) / np.log(max(brightness_score, 1) / 255.0)
    gamma = np.clip(gamma, 0.4, 2.5)  # safety bounds, avoids extreme values

    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                       for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(cv_img, table)
    return corrected, round(float(gamma), 2)


# ============================================================
# STEP 9 — Final resizing
# ============================================================
def resize_final(cv_img):
    """Resizes to a target width, preserving the aspect ratio."""
    h, w = cv_img.shape[:2]

    if w > TARGET_MAX_WIDTH:
        scale = TARGET_MAX_WIDTH / w
        new_size = (TARGET_MAX_WIDTH, int(h * scale))
        resized = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_AREA)
        return resized, "downscaled"
    elif w < MIN_ACCEPTABLE_WIDTH:
        scale = MIN_ACCEPTABLE_WIDTH / w
        new_size = (MIN_ACCEPTABLE_WIDTH, int(h * scale))
        resized = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_CUBIC)
        return resized, "upscaled"
    return cv_img, "unchanged"


# ============================================================
# STEP 10 — Final encoding (base64, for the VLM API call)
# ============================================================
def encode_to_base64(cv_img, format="PNG"):
    """Converts the final image (OpenCV BGR array) to base64 PNG, with no
    metadata."""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)  # PIL doesn't re-inject EXIF here -> implicit cleanup
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


# ============================================================
# MAIN PIPELINE — full orchestration
# ============================================================
def process_ecg_image(image_path, save_debug_path=None):
    """
    Runs the full pipeline and returns:
    - the final image encoded in base64 (ready for the VLM API)
    - a complete quality report (before/after, corrections applied)
    """
    report = {"source_path": image_path, "corrections_applied": {}}

    # 1. Loading
    pil_img = load_and_validate(image_path)

    # 2. Orientation
    pil_img, rotation_applied = correct_orientation(pil_img)
    report["corrections_applied"]["exif_rotation_deg"] = rotation_applied

    # 3. RGB
    pil_img, rgb_conversion_done = flatten_to_rgb(pil_img)
    report["corrections_applied"]["rgb_conversion"] = rgb_conversion_done

    # Convert to OpenCV array (BGR) for the rest of the processing
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 4. Deskew (best-effort)
    cv_img, skew_angle = deskew_if_needed(cv_img)
    report["corrections_applied"]["deskew_angle_deg"] = skew_angle

    # 5. Measurements BEFORE correction (blur/contrast/brightness)
    quality_before = measure_quality(cv_img)
    report["quality_before"] = quality_before

    # 6. Adaptive sharpening
    cv_img, sharpen_amount = adaptive_sharpen(cv_img, quality_before["blur_score"])
    report["corrections_applied"]["sharpen_amount"] = sharpen_amount

    # 7. Adaptive contrast
    cv_img, clahe_clip = adaptive_contrast(cv_img, quality_before["contrast"])
    report["corrections_applied"]["clahe_clip_limit"] = clahe_clip

    # 8. Adaptive brightness
    cv_img, gamma_applied = adaptive_brightness(cv_img, quality_before["brightness"])
    report["corrections_applied"]["gamma_applied"] = gamma_applied

    # 9. Final resize
    cv_img, resize_action = resize_final(cv_img)
    report["corrections_applied"]["resize_action"] = resize_action

    # Measurements AFTER correction (to verify the real effect of the corrections)
    quality_after = measure_quality(cv_img)
    report["quality_after"] = quality_after

    # Optional debug save (for visual inspection)
    if save_debug_path:
        cv2.imwrite(save_debug_path, cv_img)
        report["debug_image_saved"] = save_debug_path

    # 10. Final encoding
    encoded_b64 = encode_to_base64(cv_img)

    # ------------------------------------------------------------
    # Overall confidence level (indicative, to pass on to the VLM)
    # ------------------------------------------------------------
    residual_issues = []
    if quality_after["blur_score"] < BLUR_SEVERE_THRESHOLD:
        residual_issues.append("significant_residual_blur")
    if quality_after["contrast"] < CONTRAST_LOW_THRESHOLD * 0.7:
        residual_issues.append("contrast_still_low")

    report["residual_issues"] = residual_issues
    report["source_quality_level"] = (
        "degraded" if residual_issues else
        "corrected" if any(v not in (0, 0.0, False, "unchanged")
                            for v in report["corrections_applied"].values())
        else "clean"
    )

    return encoded_b64, report


# ============================================================
# TEST — compare Variant A vs Variant B
# ============================================================
if __name__ == "__main__":
    import json

    test_images = {
        "Variant A (clean)": "./genecg_downloaded/Dataset_A_ECGs_without_imperfections/00000/00001_hr_1R.png",
        "Variant B (degraded)": "./genecg_downloaded/Dataset_B_ECGs_with_imperfections/00000/00001_hr_1R.png",
    }

    for label, path in test_images.items():
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        try:
            _, report = process_ecg_image(
                path,
                save_debug_path=f"./debug_{label[:11].replace(' ', '_')}.png"
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {e}")


# ============================================================
# KNOWN LIMITATIONS OF THIS PIPELINE (to document in the report)
# ============================================================
"""
1. NO FULL PERSPECTIVE CORRECTION
   Only a minor skew (simple rotation < 15°) is corrected. A real photo
   taken at an angle (trapezoid distortion) is NOT straightened. This
   would require detecting the ECG paper's contour + a homographic
   transform (not implemented here for cost/time reasons).

2. DEBLURRING = APPARENT IMPROVEMENT, NOT INFORMATION RECOVERY
   Unsharp masking emphasizes existing edges but cannot reconstruct
   detail that was genuinely lost at acquisition time (unlike a deep
   learning deblurring model, which is more expensive).

3. THRESHOLDS ARE EMPIRICALLY SET, NOT CLINICALLY VALIDATED
   The thresholds (BLUR_SEVERE_THRESHOLD, CONTRAST_LOW_THRESHOLD, etc.)
   are reasonable starting values, to be adjusted after observing a
   larger sample of real images (variants A/B, and ideally real
   user-submitted photos).

4. NO NON-ECG CONTENT DETECTION (in THIS module) - now addressed upstream
   This preprocessing module itself does not verify that the image actually
   contains an ECG (it will happily deskew/sharpen any image). That "is this
   an ECG?" content check is now implemented ONE LEVEL UP, as an opt-in gate
   in the image agent: ecg_image_gate.is_ecg_image (CLIP zero-shot), wired
   into ecg_image_agent.generate_image_ecg_contribution(ecg_gate=True), which
   rejects a non-ECG upload with a "not_an_ecg_image" flag BEFORE this
   preprocessing (or the VLM/classifier) ever runs. So the guard exists; it
   just lives with the agent, not in this low-level image-quality step.

5. NO HANDLING OF STRONG ROTATIONS (>15°) OR UPSIDE-DOWN IMAGES
   WITHOUT EXIF metadata. A photo that is upside down with no EXIF data
   will not be detected as such (no semantic content detection).

6. DESKEW CAN FAIL ON VERY NOISY IMAGES
   Contour detection (used for skew) assumes a minimal contrast between
   the ECG trace and the background. On an extremely degraded image,
   this step may detect nothing and correct nothing (fail-safe
   behavior: do nothing rather than over-correct).
"""
