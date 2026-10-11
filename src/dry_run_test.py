"""
Dry-run test: no Ollama call, no image download. Verifies that
build_image_findings_prompt() produces a well-formed prompt with per-lead
evidence keys, and that parse_findings_response() accepts a simulated
response citing a specific lead (e.g. ecg_image_V2).
"""
from ecg_image_agent import (
    build_image_findings_prompt,
    ALL_IMAGE_EVIDENCE_KEYS,
    LEAD_EVIDENCE_KEYS,
    IMAGE_EVIDENCE_KEY,
)
from agent_common import parse_findings_response
import json

# Fake quality features (as if a real image had been processed)
fake_quality_features = {
    "image_width_px": 1800,
    "image_height_px": 950,
    "blur_score_before": 300.0,
    "blur_score_after": 300.0,
    "contrast_before": 40.0,
    "contrast_after": 40.0,
    "brightness_before": 130.0,
    "brightness_after": 130.0,
    "sharpen_amount_applied": 0.0,
    "clahe_clip_limit_applied": 0.0,
    "gamma_applied": 1.0,
    "deskew_angle_deg": 0.0,
    "resize_action": "unchanged",
    "residual_issue_count": 0,
    "source_quality_level": "clean",
}

fake_summary = "- Image size after preprocessing: 1800x950 px (unchanged)\n- Overall source quality assessment: clean"

# 1. Build the prompt and inspect it
prompt = build_image_findings_prompt(fake_summary, fake_quality_features)
print("=" * 70)
print("GENERATED PROMPT (first 2000 chars)")
print("=" * 70)
print(prompt[:2000])
print("\n[...]\n")

assert "ecg_image_V2" in prompt, "Lead-specific key V2 missing from prompt!"
assert "ecg_image_overall" in prompt, "Overall key missing from prompt!"
assert prompt.count("Image ") >= 1, "No image section found!"
print("PASS: prompt contains per-lead keys and overall key.\n")

# 2. Simulate a model response citing a SPECIFIC lead
fake_llm_response = json.dumps({
    "findings": [
        {
            "feature": "st_elevation_suspected",
            "value": "present",
            "rationale": "Visible ST segment elevation in this lead's strip.",
            "confidence": 0.7,
            "evidence_ref": "ecg_image_V2"
        },
        {
            "feature": "rhythm_regularity",
            "value": "regular",
            "rationale": "R-R intervals appear visually consistent across the tracing.",
            "confidence": 0.6,
            "evidence_ref": "ecg_image_overall"
        }
    ],
    "flags_for_review": []
})

findings, flags = parse_findings_response(
    fake_llm_response,
    valid_evidence_keys=set(fake_quality_features.keys()) | ALL_IMAGE_EVIDENCE_KEYS,
    image_evidence_keys=ALL_IMAGE_EVIDENCE_KEYS,
)

print("=" * 70)
print("PARSED FINDINGS")
print("=" * 70)
for f in findings:
    print(f"  feature={f.feature!r} evidence_ref={f.evidence_ref!r} evidence_type={f.evidence_type!r}")

assert findings[0].evidence_ref == "ecg_image_V2"
assert findings[0].evidence_type == "image_region"
assert findings[1].evidence_ref == "ecg_image_overall"
print("\nPASS: lead-specific citation parses correctly, evidence_type set properly.")

# 3. Simulate a model response citing an INVALID lead (should fail)
bad_response = json.dumps({
    "findings": [{
        "feature": "x", "value": "y", "rationale": "z",
        "confidence": 0.5, "evidence_ref": "ecg_image_V13"  # doesn't exist
    }],
    "flags_for_review": []
})
try:
    parse_findings_response(
        bad_response,
        valid_evidence_keys=set(fake_quality_features.keys()) | ALL_IMAGE_EVIDENCE_KEYS,
        image_evidence_keys=ALL_IMAGE_EVIDENCE_KEYS,
    )
    print("FAIL: invalid lead key was NOT rejected!")
except Exception as e:
    print(f"PASS: invalid lead key correctly rejected ({type(e).__name__}).")
