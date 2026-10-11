"""
Sanity check: does the multimodal model actually SEE the attached image,
or is it silently ignoring it and just generating plausible-sounding text?

Strategy: ask simple, independently-verifiable visual questions - NOT the
full clinical JSON schema - so a wrong/evasive answer clearly indicates a
real ingestion problem rather than a clinical reasoning limitation.

Ground truth used here: any standard 12-lead ECG image (which is what
GenECG produces) always shows exactly 12 distinct lead traces/labels.
This is known independently of the model's answer, so it's a clean check.
"""

import sys
from genecg_fixtures import fetch_test_image
from adaptive_ecg_pipeline import process_ecg_image
from agent_common import generate_report_medgemma

ECG_ID = 1
VARIANT = "B"

# Compare the third-party ECG-specific GGUF against the OFFICIAL Google
# MedGemma 4B (multimodal, on Ollama's official library) to isolate
# whether image-ignoring is a conversion/integration bug specific to the
# third-party GGUF, or a broader model-capability limitation. The official
# model was NOT trained on ECG images (chest X-ray/dermatology/ophthalmology/
# histopathology only) so its CLINICAL answers are expected to be weaker -
# this test only checks whether it "sees" the image at all, not whether its
# ECG readings are any good.
#
# Usage:
#   python sanity_check_image_ingestion.py                 # third-party GGUF (default)
#   python sanity_check_image_ingestion.py medgemma:4b      # official Google model
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else \
    "hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M"

# Plain-language questions, no JSON schema, no clinical framing -
# just testing raw visual perception.
SANITY_QUESTIONS = [
    "How many separate lead traces (strips) are visible in this ECG image? "
    "Just answer with a number and one short sentence.",

    "Is there a background grid (small squares) visible behind the ECG "
    "traces in this image? Answer yes or no, with one short sentence.",

    "List the lead labels you can actually read printed on this image "
    "(e.g. I, II, III, aVR...). If you cannot read any labels clearly, "
    "say so explicitly instead of guessing.",
]

image_path = fetch_test_image(ECG_ID, VARIANT)
b64_image, quality_report = process_ecg_image(image_path)

print(f"Image: {image_path}")
print(f"Preprocessing quality level: {quality_report['source_quality_level']}\n")

for i, question in enumerate(SANITY_QUESTIONS, start=1):
    print("=" * 70)
    print(f"QUESTION {i}: {question}")
    print("=" * 70)

    # WITH the image attached
    answer_with_image = generate_report_medgemma(
        question, model=MODEL_TAG, images=[b64_image], num_predict=200,
    )
    print(f"[WITH image]:\n{answer_with_image}\n")

    # WITHOUT the image (control) - if the answer is suspiciously similar,
    # that's strong evidence the image isn't actually influencing the output.
    answer_without_image = generate_report_medgemma(
        question, model=MODEL_TAG, images=None, num_predict=200,
    )
    print(f"[WITHOUT image, control]:\n{answer_without_image}\n")

    if answer_with_image.strip() == answer_without_image.strip():
        print("!!! IDENTICAL output with and without the image - "
              "strong sign the image is being ignored.\n")
    else:
        print("(Outputs differ - image likely has SOME influence; "
              "still check the content above for correctness.)\n")

print("\n" + "=" * 70)
print("EXPECTED (ground truth, independent of the model):")
print("=" * 70)
print("- 12 distinct lead traces (standard 12-lead ECG)")
print("- Background grid: check visually - GenECG images typically include one")
print("- Lead labels should include some subset of: I, II, III, aVR, aVL, aVF, "
      "V1, V2, V3, V4, V5, V6")
print("\nCompare the model's answers above to this ground truth to judge "
      "whether it is genuinely reading the image.")
