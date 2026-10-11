# archive/

Non-runtime artifacts moved here during a cleanup pass (2026-08-22). **None of
these are read by the app** (`streamlit run Home.py`) or by the validation page —
they are historical outputs and can be reviewed, kept, or deleted at leisure.

Moved from `results/`:

| Item | What it is |
|---|---|
| `mri_pipeline_training_log.txt` (806 KB), `mri_pipeline_training_err.txt`, `mri_classifier_training_log.txt` | raw training logs of the YOLO/U-Net/classifier runs |
| `cot_reliability_raw/`, `cot_repeated_sample_raw/`, `cot_reliability_history/`, `temp_seed_test_raw/` | raw per-generation dumps from the CoT / temperature-seed reliability experiments |
| `rag_knowledge_base_clinical/` | the early "dirty/jargon" RAG index, superseded by the clean production index in `results/rag_knowledge_base/` |
| `mri_roi_*.png`, `mri_sanity_check_*.png`, `mri_shared_bbox_*.png` | early MRI ROI / sanity-check figures |
| `comparison_result_ptbxl_kaggle_vs_physionet.json` | one-off Kaggle-vs-PhysioNet integrity check result |

Second pass — additional historical `results/` outputs (also not read by the app):

| Item | What it is |
|---|---|
| `sep_agent_demo/`, `sep_agent_demo_v2_with_mri_classifier/`, `sep_agent_3modality_test/`, `patient_data_agent_test/` | one-off Sep-Agent / patient-agent demo & test outputs |
| `evaluation/`, `evaluation_with_embedding/` | older report-evaluation runs (the cited run, `evaluation_brief/`, was kept in `results/`) |
| `records/` | early single-record ECG pipeline figures |
| `cot_reliability_test_after_fix.json`, `cot_repeated_reliability_sample.json`, `temp_seed_test_*.json`, `rag_embedding_comparison.json`/`.pdf`, `rag_pearson_evaluation.json` | loose experiment result summaries |

Kept in `results/` (still referenced by CLAUDE.md as the authoritative write-ups, or read by
the app): `ecg_multilead_validation/`, `ecg_territorial_validation/`, `kg_integration/`,
`rag_ab_test/`, `report_package/`, `evaluation_brief/`, the model/validation folders, and the
`*.md` result docs.

Also removed during the same pass (regenerable, not archived): every
`__pycache__/` directory.

The project's own pre-existing archives (`results/_archive/`, `results/_backups/`)
were left untouched.
