"""
Debug helper - prints the RAW text response from the image LLM, with no
parsing/validation, so you can see exactly what's malformed when
generate_image_ecg_contribution raises FindingsParseError.

Usage:
    python debug_llm_raw_response.py --ecg_id 1 --variant B
"""

import argparse

from genecg_fixtures import fetch_test_image
from adaptive_ecg_pipeline import process_ecg_image
from ecg_image_agent import (
    compute_image_quality_features,
    build_image_clinical_summary,
    build_image_findings_prompt,
    _call_with_image_fallback,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg_id", type=int, default=1)
    parser.add_argument("--variant", choices=["A", "B"], default="B")
    args = parser.parse_args()

    image_path = fetch_test_image(args.ecg_id, args.variant)
    print(f"Image: {image_path}\n")

    b64_image, quality_report = process_ecg_image(image_path)
    quality_features = compute_image_quality_features(quality_report)
    clinical_summary = build_image_clinical_summary(quality_report)
    prompt = build_image_findings_prompt(clinical_summary, quality_features)

    print("=" * 70)
    print("PROMPT SENT TO THE MODEL")
    print("=" * 70)
    print(prompt)

    used_model_holder = {"tag": None}
    raw_response = _call_with_image_fallback(prompt, b64_image, used_model_holder)

    print("\n" + "=" * 70)
    print(f"RAW RESPONSE (model actually used: {used_model_holder['tag']})")
    print("=" * 70)
    print(raw_response)
    print("=" * 70)
    print(f"Response length: {len(raw_response)} characters, "
          f"{raw_response.count(chr(10)) + 1} lines")

    # Point directly at the character/line json.loads chokes on, same as
    # the real parser would report.
    import json
    try:
        json.loads(raw_response)
        print("\n-> This IS valid JSON on its own (the failure must be a "
              "schema/field validation issue inside parse_findings_response, "
              "not a raw JSON syntax issue).")
    except json.JSONDecodeError as e:
        print(f"\n-> json.loads fails here: {e}")
        lines = raw_response.split("\n")
        line_no = e.lineno - 1
        print("   Context around the failure:")
        for i in range(max(0, line_no - 2), min(len(lines), line_no + 3)):
            marker = ">>>" if i == line_no else "   "
            print(f"   {marker} line {i+1}: {lines[i]!r}")


if __name__ == "__main__":
    main()
