"""
Download a class-stratified set of GenECG images (variant A) and build a
manifest (ecg_id, local_path, 5-superclass multi-hot label, split) for training
the GenECG image classifier. Resumable: cached images are skipped instantly, so
re-running after an interruption just continues.

Train split = PTB-XL folds 1-9, test split = fold 10 (patient-disjoint).
Outputs: results/ecg_image_pipeline/genecg_classifier_manifest.csv
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GENECG_DOWNLOAD_DIR",
                      os.path.join(os.getcwd(), "genecg_downloaded"))
from ecg_embedding import load_scp_superclass_map, scp_codes_to_multihot, DIAGNOSTIC_SUPERCLASSES as CLS
from genecg_fixtures import fetch_test_image

ROOT = "kaggle/archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
MAPPING = "src/ptbxl_with_genecg_paths.csv"
OUT = "results/ecg_image_pipeline"
MANIFEST = f"{OUT}/genecg_classifier_manifest.csv"
VARIANT = "A"
PER_CLASS_TRAIN = 180     # per superclass, folds 1-9
PER_CLASS_TEST = 45       # per superclass, fold 10
SEED = 0
os.makedirs(OUT, exist_ok=True)

code_map = load_scp_superclass_map(f"{ROOT}/scp_statements.csv")
df = pd.read_csv(MAPPING).dropna(subset=["scp_codes"]).reset_index(drop=True)
labels = np.stack(df["scp_codes"].apply(lambda s: scp_codes_to_multihot(s, code_map)).to_numpy())
folds = df["strat_fold"].to_numpy()
rng = np.random.RandomState(SEED)

def sample(mask, per_class):
    chosen = []
    for ci in range(len(CLS)):
        pos = np.nonzero(mask & (labels[:, ci] == 1))[0]
        if len(pos):
            chosen.extend(rng.choice(pos, size=min(per_class, len(pos)), replace=False).tolist())
    return list(dict.fromkeys(chosen))

train_idx = sample(folds <= 9, PER_CLASS_TRAIN)
test_idx = sample(folds == 10, PER_CLASS_TEST)
plan = [(i, "train") for i in train_idx] + [(i, "test") for i in test_idx]
print(f"planned: {len(train_idx)} train + {len(test_idx)} test = {len(plan)} images")

rows, t0, ok = [], time.time(), 0
for k, (idx, split) in enumerate(plan):
    row = df.iloc[idx]
    ecg_id = int(row["ecg_id"])
    try:
        path = fetch_test_image(ecg_id, VARIANT)
        rows.append({"ecg_id": ecg_id, "path": path, "split": split,
                     **{c: int(labels[idx, i]) for i, c in enumerate(CLS)}})
        ok += 1
        if (k + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(MANIFEST, index=False)   # checkpoint
            print(f"  {k+1}/{len(plan)} ok={ok} ({time.time()-t0:.0f}s, {(time.time()-t0)/ok:.2f}s/img)")
    except Exception as e:
        print(f"  {k+1}/{len(plan)} ecg_id={ecg_id} SKIP {type(e).__name__}: {e}")

pd.DataFrame(rows).to_csv(MANIFEST, index=False)
print(f"\nDONE: {ok}/{len(plan)} downloaded -> {MANIFEST} ({time.time()-t0:.0f}s)")
