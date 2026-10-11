"""
Duration-augmentation experiment: does training with RANDOM-LENGTH crops make
the ECG embedding models robust to variable input DURATION (10 / 5 / 2 s),
instead of being locked to PTB-XL's fixed 10 s records?

Motivation (supervisor's point): a model that only ever sees 10 s at training
time stays tied to PTB-XL's format. The fix is not "RNN instead of CNN" but
training-time length augmentation - which helps BOTH architectures. This script
fine-tunes the existing stratfold checkpoints (CNN = ECGEmbeddingNet,
GRU = ECGGRUNet) with random-duration crops, then re-runs the 10/5/2 s
truncation test and compares against the un-augmented baselines.

Standalone experiment (does NOT modify ecg_embedding.py / ecg_agent.py). Uses
the same patient-disjoint strat_fold split. Signals are pre-loaded into RAM once
(disk wfdb reads are the bottleneck) and cached to an .npz.

Run:  python src/ecg_duration_augmentation_experiment.py
Outputs: results/ecg_embedding/duration_augmentation/
"""
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from ecg_embedding import (
    PTBXLDataset, ECGEmbeddingNet, ECGGRUNet,
    load_model_checkpoint, save_model_checkpoint,
    DIAGNOSTIC_SUPERCLASSES as CLS,
)

ROOT = "kaggle/archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT = "results/ecg_embedding/duration_augmentation"
CACHE = f"{OUT}/preloaded_signals.npz"
os.makedirs(OUT, exist_ok=True)

# Sizes kept modest so the one-time disk preload is tractable; this is a
# proof-of-concept fine-tune, not a full production retrain.
N_TRAIN, N_VAL, N_TEST = 2000, 600, 600
EPOCHS = 8
LR = 5e-4
BATCH = 32
FULL_LEN = 1000            # 10 s @ 100 Hz (lr)
MIN_CROP = 200             # 2 s  @ 100 Hz
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"device={device}")


def znorm(x, axis):
    return (x - x.mean(axis=axis, keepdims=True)) / (x.std(axis=axis, keepdims=True) + 1e-8)


# ----------------------------------------------------------------------
# 1. Preload signals into RAM (once), cache to .npz
# ----------------------------------------------------------------------
def preload():
    if os.path.exists(CACHE):
        print(f"loading cached signals from {CACHE}")
        d = np.load(CACHE)
        return (d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], d["Xte"], d["Yte"])

    ds = PTBXLDataset(f"{ROOT}/ptbxl_database.csv", f"{ROOT}/scp_statements.csv", ROOT, resolution="lr")
    tr, va, te = ds.strat_fold_indices()
    rng = np.random.RandomState(SEED)
    tr = rng.choice(tr, size=N_TRAIN, replace=False)
    va = rng.choice(va, size=N_VAL, replace=False)
    te = rng.choice(te, size=N_TEST, replace=False)

    def grab(idxs, tag):
        X, Y = [], []
        t0 = time.time()
        for k, i in enumerate(idxs):
            sig, lab = ds[int(i)]                 # (12,1000) z-normed full, label (5,)
            X.append(sig.numpy().astype("float32"))
            Y.append(lab.numpy().astype("float32"))
            if (k + 1) % 200 == 0:
                print(f"  {tag}: {k+1}/{len(idxs)} ({time.time()-t0:.0f}s)")
        return np.stack(X), np.stack(Y)

    Xtr, Ytr = grab(tr, "train"); Xva, Yva = grab(va, "val"); Xte, Yte = grab(te, "test")
    np.savez_compressed(CACHE, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte)
    print(f"cached to {CACHE}")
    return Xtr, Ytr, Xva, Yva, Xte, Yte


# ----------------------------------------------------------------------
# 2. Length-augmented fine-tuning (random crop LENGTH + START per batch)
# ----------------------------------------------------------------------
def finetune(model, Xtr, Ytr, Xva, Yva, name):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    n = len(Xtr_t)
    best_val = float("inf"); best_path = f"{OUT}/{name}.best.pt"
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for s in range(0, n, BATCH):
            bidx = perm[s:s + BATCH]
            # one random crop window for the whole batch (keeps tensors stackable)
            L = random.randint(MIN_CROP, FULL_LEN)
            start = random.randint(0, FULL_LEN - L)
            xb = Xtr_t[bidx][:, :, start:start + L]
            xb = znorm(xb, axis=2).to(device)     # re-normalize over the crop
            yb = Ytr_t[bidx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            run += loss.item() * len(bidx)
        # validation at full length (proxy)
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xva).to(device); yv = torch.from_numpy(Yva).to(device)
            vloss = crit(model(xv), yv).item()
        print(f"  [{name}] epoch {ep+1}/{EPOCHS} train {run/n:.4f} val {vloss:.4f}")
        if vloss < best_val:
            best_val = vloss
            save_model_checkpoint(model, best_path)
    load_model_checkpoint(model, best_path, device=device)
    return model


# ----------------------------------------------------------------------
# 3. Truncation eval (10 / 5 / 2 s), macro F1 at threshold 0.5
# ----------------------------------------------------------------------
DURATIONS = [(1000, "10s"), (500, "5s"), (200, "2s")]


def per_class_f1(y_true, y_pred):
    out = []
    for c in range(y_true.shape[1]):
        tp = np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 1))
        fp = np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 0))
        fn = np.sum((y_pred[:, c] == 0) & (y_true[:, c] == 1))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.array(out)


def eval_durations(model, Xte, Yte):
    model = model.to(device)
    model.eval()
    res = {}
    with torch.no_grad():
        for N, tag in DURATIONS:
            seg = znorm(Xte[:, :, :N], axis=2).astype("float32")
            x = torch.from_numpy(seg).to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            res[tag] = per_class_f1(Yte, pred)
    return res


def main():
    Xtr, Ytr, Xva, Yva, Xte, Yte = preload()
    print(f"shapes: train {Xtr.shape} val {Xva.shape} test {Xte.shape}")

    lines = ["# Duration-augmentation experiment\n",
             f"Fine-tuned from stratfold checkpoints with random-length crops "
             f"({MIN_CROP//100}-{FULL_LEN//100}s), {EPOCHS} epochs, "
             f"train/val/test = {N_TRAIN}/{N_VAL}/{N_TEST} (subsampled), seed {SEED}.\n",
             "\nMacro F1 by input duration (threshold 0.5), BEFORE (baseline "
             "checkpoint) vs AFTER (length-augmented fine-tune):\n",
             "\n| model | variant | 10s | 5s | 2s | drop 10->2s |",
             "|---|---|---:|---:|---:|---:|"]

    specs = [
        ("CNN", ECGEmbeddingNet, "results/ecg_embedding/embedding_stratfold.best.pt", "cnn_stratfold_lenaug"),
        ("GRU", ECGGRUNet,       "results/ecg_embedding/gru_stratfold.best.pt",       "gru_stratfold_lenaug"),
    ]
    percls = ["\n## Per-class F1 at 2s (after augmentation)\n",
              "\n| model | " + " | ".join(CLS) + " |", "|---|" + "---|" * len(CLS)]

    for label, cls, base_ckpt, aug_name in specs:
        # baseline
        base = cls(); load_model_checkpoint(base, base_ckpt, device=device)
        b = eval_durations(base, Xte, Yte)
        bm = {t: b[t].mean() for t in ("10s", "5s", "2s")}
        lines.append(f"| {label} | baseline | {bm['10s']:.3f} | {bm['5s']:.3f} | "
                     f"{bm['2s']:.3f} | -{bm['10s']-bm['2s']:.3f} |")
        # length-augmented fine-tune
        aug = cls(); load_model_checkpoint(aug, base_ckpt, device=device)
        print(f"\n>>> fine-tuning {label} with length augmentation")
        aug = finetune(aug, Xtr, Ytr, Xva, Yva, aug_name)
        a = eval_durations(aug, Xte, Yte)
        am = {t: a[t].mean() for t in ("10s", "5s", "2s")}
        lines.append(f"| {label} | **length-aug** | {am['10s']:.3f} | {am['5s']:.3f} | "
                     f"**{am['2s']:.3f}** | -{am['10s']-am['2s']:.3f} |")
        percls.append(f"| {label} length-aug | " + " | ".join(f"{v:.3f}" for v in a['2s']) + " |")

    report = "\n".join(lines + percls) + "\n"
    with open(f"{OUT}/duration_augmentation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"saved -> {OUT}/duration_augmentation_report.md")


if __name__ == "__main__":
    main()
