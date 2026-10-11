"""
Verification / smoke test for the image ECG pipeline, in three layers:
  1. adaptive_ecg_pipeline.process_ecg_image() alone (preprocessing only)
  2. ecg_image_agent.generate_image_ecg_contribution() (full agent, calls Ollama)
  3. ecg_router.route_ecg_upload() (dispatch + "always respond" guarantees)

Replaces check_image_quality.py + download_genecg_sample.py (merged, single
set of thresholds - the ones adaptive_ecg_pipeline.py actually uses in prod,
not a second divergent copy).

Usage:
    python verify_image_pipeline.py                  # runs everything, ecg_id=1
    python verify_image_pipeline.py --ecg_id 42
    python verify_image_pipeline.py --skip_llm        # layers 1 + router failure
                                                       # modes only, no Ollama call
"""

import argparse
import os
import sys

from genecg_fixtures import fetch_test_image

MAPPING_CSV = "ptbxl_with_genecg_paths.csv"   # relative path - adjust if needed
LOCAL_DOWNLOAD_DIR = "./genecg_downloaded"
HF_REPO_ID = "edcci/GenECG"


def _section(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ----------------------------------------------------------------------
# Layer 1 - preprocessing only, no LLM, no agent
# ----------------------------------------------------------------------
def check_layer1_preprocessing(image_path: str, label: str):
    from adaptive_ecg_pipeline import process_ecg_image

    _section(f"LAYER 1 - preprocessing only : {label}")
    b64_image, report = process_ecg_image(image_path)

    assert isinstance(b64_image, str) and len(b64_image) > 0, "empty base64 output"
    assert report["source_quality_level"] in ("clean", "corrected", "degraded")
    print(f"  quality level: {report['source_quality_level']}")
    print(f"  residual issues: {report['residual_issues']}")
    print(f"  corrections applied: {report['corrections_applied']}")
    print("  PASS")
    return report


# ----------------------------------------------------------------------
# Layer 2 - full agent (needs Ollama + medgemma-ecginstruct:4b running)
# ----------------------------------------------------------------------
def check_layer2_agent(image_path: str, label: str):
    from ecg_image_agent import generate_image_ecg_contribution, IMAGE_EVIDENCE_KEY

    _section(f"LAYER 2 - full agent (calls Ollama) : {label}")
    contribution = generate_image_ecg_contribution(image_path, record_id=f"test_{label}")

    assert contribution.agent_id == "ecg_image"
    assert isinstance(contribution.findings, list)
    print(f"  {len(contribution.findings)} finding(s), "
          f"{len(contribution.flags_for_review)} flag(s): {contribution.flags_for_review}")

    image_grounded = [f for f in contribution.findings if f.evidence_ref == IMAGE_EVIDENCE_KEY]
    print(f"  {len(image_grounded)} finding(s) grounded in the image itself "
          f"(evidence_type=image_region)")
    if not image_grounded:
        print("  WARNING: no finding cited the image as evidence - the model may not "
              "actually be looking at it (check the multimodal import, see "
              "setup_medgemma_ecginstruct_ollama.sh's sanity-check note about silent "
              "image-ignoring failures).")

    for f in contribution.findings:
        assert 0.0 <= f.confidence <= 1.0, f"confidence out of range: {f}"
    print(f"  model used: {contribution.trace.get('model')}")
    print("  PASS" if image_grounded else "  PASS (with warning above)")
    return contribution


# ----------------------------------------------------------------------
# Layer 3 - router, including its "always respond" failure modes
# ----------------------------------------------------------------------
def check_layer3_router(image_path: str, skip_llm: bool):
    from ecg_router import route_ecg_upload

    _section("LAYER 3 - router dispatch")

    if not skip_llm:
        contribution = route_ecg_upload(image_path, record_id="test_router_ok")
        assert contribution.agent_id == "ecg_image"
        print(f"  valid image -> agent_id={contribution.agent_id}: PASS")

    # --- failure mode 1: unrecognized extension ---
    fake_path = "not_an_ecg.xyz"
    open(fake_path, "w").close()
    contribution = route_ecg_upload(fake_path, record_id="test_bad_ext")
    assert contribution.agent_id == "ecg_router"
    assert "unrecognized_upload_type" in contribution.flags_for_review
    os.remove(fake_path)
    print("  unrecognized extension -> caught, still returned a contribution: PASS")

    # --- failure mode 2: corrupted/unreadable image ---
    corrupt_path = "corrupt.png"
    with open(corrupt_path, "wb") as f:
        f.write(b"not a real png")
    contribution = route_ecg_upload(corrupt_path, record_id="test_corrupt")
    assert "image_unreadable" in contribution.flags_for_review
    os.remove(corrupt_path)
    print("  corrupted image -> caught (image_unreadable), still returned: PASS")

    # --- failure mode 3: nonexistent file ---
    contribution = route_ecg_upload("/tmp/does_not_exist_at_all.png", record_id="test_missing")
    assert contribution.findings == []
    assert "agent_error" in contribution.flags_for_review or "image_unreadable" in contribution.flags_for_review
    print("  missing file -> caught, still returned: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg_id", type=int, default=1)
    parser.add_argument("--variant", choices=["A", "B"], default="B",
                         help="B (with imperfections) exercises the preprocessing more; "
                              "A is the clean baseline.")
    parser.add_argument("--skip_llm", action="store_true",
                         help="Skip layers that call Ollama (layer 1 + router failure "
                              "modes only) - use this if Ollama isn't running yet.")
    args = parser.parse_args()

    image_path = fetch_test_image(args.ecg_id, args.variant)
    label = f"ecg_id={args.ecg_id} variant={args.variant}"
    print(f"Test image: {image_path}")

    report = check_layer1_preprocessing(image_path, label)

    if not args.skip_llm:
        check_layer2_agent(image_path, label)

    check_layer3_router(image_path, skip_llm=args.skip_llm)

    _section("ALL LAYERS PASSED" if not args.skip_llm else "LAYER 1 + ROUTER FAILURE MODES PASSED")


if __name__ == "__main__":
    sys.exit(main())
