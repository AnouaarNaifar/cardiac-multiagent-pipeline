"""
GenECG image classifier: a discriminative CNN trained on GenECG ECG IMAGES ->
PTB-XL 5-superclass multi-hot labels. The image-branch analogue of
ecg_embedding.py's signal classifier: it outputs TRUE per-class probabilities
(unlike a VLM's self-reported confidence) and is validated on a patient-disjoint
held-out split (PTB-XL fold 10), directly comparable to the 100% VLM zero-shot
baseline (macro F1 ~0.31, MI 0.00).

Uses NO signal code and NO digitization - pure image -> CNN -> probabilities,
compatible with the direct-image design.

Reads results/ecg_image_pipeline/genecg_classifier_manifest.csv (built by
genecg_classifier_download.py). Requires torch (+ torchvision if available; falls
back to a small from-scratch CNN otherwise).
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from ecg_embedding import DIAGNOSTIC_SUPERCLASSES as CLS

OUT = "results/ecg_image_pipeline"
MANIFEST = f"{OUT}/genecg_classifier_manifest.csv"
IMG_SIZE = 384
BATCH = 16
MAX_EPOCHS = 40
PATIENCE = 5            # early-stopping patience on the validation split
VAL_FRACTION = 0.15     # carved from the train pool; the test fold stays held-out
LR = 1e-4
SEED = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class GenECGDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(r["path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.train and np.random.rand() < 0.5:      # mild brightness jitter
            x = np.clip(x * np.random.uniform(0.85, 1.15), 0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))
        y = torch.tensor([float(r[c]) for c in CLS])
        return x, y


def build_model():
    """Pretrained ResNet18 if torchvision is available; else a small CNN."""
    try:
        import torchvision
        try:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            print("using torchvision ResNet18 (ImageNet pretrained)")
        except Exception as e:
            m = torchvision.models.resnet18(weights=None)
            print(f"using torchvision ResNet18 (random init - weights download failed: {e})")
        m.fc = nn.Linear(m.fc.in_features, len(CLS))
        return m
    except Exception as e:
        print(f"torchvision unavailable ({e}); using small from-scratch CNN")
        return SmallCNN()


class SmallCNN(nn.Module):
    def __init__(self, n=len(CLS)):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                            nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.f(x))


# ----------------------------------------------------------------------
# Inference entry points (importable by ecg_image_agent.py; no training,
# no ImageNet-weights download - the architecture is rebuilt empty and the
# saved GenECG checkpoint is loaded into it).
# ----------------------------------------------------------------------
CHECKPOINT_PATH = f"{OUT}/genecg_classifier.pt"


def load_genecg_classifier(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    """
    Load the trained GenECG image classifier for inference. Rebuilds a bare
    ResNet18 head (no ImageNet download) and loads the saved weights. Returns
    the model in eval mode on `device`.
    """
    import torchvision
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLS))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def _preprocess_image(image):
    """PIL image or path -> normalized (1, 3, IMG_SIZE, IMG_SIZE) tensor,
    identical to the training transform (resize + ImageNet normalization)."""
    if isinstance(image, str):
        image = Image.open(image)
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def predict_pathology_probabilities(image, model, device="cpu"):
    """
    Per-class probabilities for a single ECG image, via the trained classifier
    (sigmoid over the 5 multi-label logits). Mirrors
    ecg_embedding.predict_diagnostic_probabilities for the image branch.

    Inputs:
        image: path to an ECG image, or a PIL.Image.
        model: a model returned by load_genecg_classifier.
        device: "cpu" or "cuda".

    Output:
        dict "pathology_<CLASS>_probability" -> float in [0, 1], one per class
        in DIAGNOSTIC_SUPERCLASSES order.
    """
    x = _preprocess_image(image).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[0]
    return {f"pathology_{c}_probability": float(p) for c, p in zip(CLS, probs)}


def prf(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, int((yt == 1).sum())


def main():
    import copy
    df = pd.read_csv(MANIFEST)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    tr_all, te = df[df["split"] == "train"], df[df["split"] == "test"]

    # Carve a validation split from the TRAIN pool for early stopping; the
    # fold-10 test split stays fully held-out (never used for stopping).
    tr_all = tr_all.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(len(tr_all) * VAL_FRACTION))
    val, tr = tr_all.iloc[:n_val], tr_all.iloc[n_val:]
    print(f"device={device} | train={len(tr)} val={len(val)} test={len(te)} images")
    Y = tr[CLS].to_numpy()
    print("train class counts:", {c: int(Y[:, i].sum()) for i, c in enumerate(CLS)})

    train_loader = DataLoader(GenECGDataset(tr, train=True), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(GenECGDataset(val), batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader = DataLoader(GenECGDataset(te), batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model().to(device)
    # class-imbalance weighting
    pos = Y.sum(0); neg = len(Y) - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 0.5, 5.0), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def val_loss():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                tot += crit(model(x), y).item() * len(x)
        return tot / max(len(val), 1)

    # Early stopping on validation loss, restoring the best-val weights.
    best_val, best_state, best_ep, no_improve = float("inf"), None, 0, 0
    t0 = time.time()
    for ep in range(MAX_EPOCHS):
        model.train(); run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            run += loss.item() * len(x)
        vl = val_loss()
        print(f"  epoch {ep+1}/{MAX_EPOCHS} train {run/len(tr):.4f} val {vl:.4f} ({time.time()-t0:.0f}s)")
        if vl < best_val - 1e-4:
            best_val, best_state, best_ep, no_improve = vl, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stopping at epoch {ep+1} (best val {best_val:.4f} @ epoch {best_ep})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"restored best epoch {best_ep} (val {best_val:.4f})")

    # evaluate
    model.eval(); probs, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy()); trues.append(y.numpy())
    probs = np.concatenate(probs); trues = np.concatenate(trues); preds = (probs >= 0.5).astype(int)

    lines = ["# GenECG image classifier - validation (real probabilities, held-out fold 10)\n",
             f"ResNet18 (pretrained) on {len(tr)} GenECG train images (+{len(val)} val for "
             f"early stopping, best epoch {best_ep}), tested on {len(te)} held-out fold-10 "
             f"images. Multi-label, threshold 0.5. Compare to the 100% VLM zero-shot baseline "
             f"(macro F1 0.31, MI 0.00).\n",
             "\n| class | precision | recall | F1 | support |",
             "|---|---:|---:|---:|---:|"]
    macro = []
    for i, c in enumerate(CLS):
        p, r, f, sup = prf(trues[:, i], preds[:, i])
        macro.append(f)
        lines.append(f"| {c} | {p:.3f} | {r:.3f} | {f:.3f} | {sup} |")
    lines.append(f"| **macro** | | | **{np.mean(macro):.3f}** | |")
    em = float((trues == preds).all(axis=1).mean())
    lines.append(f"\nExact-match (all 5 right): **{em:.3f}** (n={len(te)})\n")
    report = "\n".join(lines) + "\n"
    open(f"{OUT}/genecg_classifier_validation.md", "w", encoding="utf-8").write(report)
    torch.save({"model_state_dict": model.state_dict(), "classes": CLS},
               f"{OUT}/genecg_classifier.pt")
    print("\n" + report + f"saved -> {OUT}/genecg_classifier_validation.md (+ .pt)")


if __name__ == "__main__":
    main()
