"""
ECG upload ingestion + guardrails (Phase P2 of the UI).

Additive, standalone, UI-agnostic (no Streamlit import). Turns an uploaded ECG
- WFDB (.hea + .dat) or a CSV of samples - into the SAME canonical form the rest
of the pipeline already expects downstream, and validates it:

    canonical signal: numpy (n_samples, n_leads) float, in REAL millivolts,
                      leads ordered as ecg_ingest.PTBXL_LEAD_ORDER when possible;
    fs:               sampling frequency in Hz (explicit, never assumed).

Why a guardrail layer at all: several downstream features depend *silently* on
these invariants, so a bad upload would produce a wrong-but-confident output:
  - the multi-lead voltage criteria (Sokolow-Lyon LVH, territorial Q) are only
    meaningful in real mV and in the right lead order;
  - the learned embedding needs all 12 leads in PTBXL order (it re-normalizes
    per-lead z-score, so it is UNIT-invariant, but NOT order/lead-count invariant);
  - the single-lead rate/rhythm pipeline just needs one lead + fs.
`validate()` therefore returns not only hard errors / soft warnings but also an
`enabled` map saying which downstream feature families are safe to compute, so
the caller can DISABLE (not just warn about) whatever would be invalid.

Dependencies: numpy, wfdb (already project deps). pandas only for CSV.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Canonical PTB-XL 12-lead order (verified against a real header).
PTBXL_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"]
_PTBXL_SET = set(PTBXL_LEAD_ORDER)

# Plausible physiological amplitude band (mV) for a resting ECG, used only to
# guess whether the declared unit is wrong (e.g. uV passed as mV -> ~1000x).
_MV_PLAUSIBLE_LOW = 0.05
_MV_PLAUSIBLE_HIGH = 10.0


@dataclass
class IngestedECG:
    """Canonical, downstream-ready representation of an uploaded ECG."""
    signal: np.ndarray            # (n_samples, n_leads), float, millivolts
    fs: float                     # Hz
    lead_names: List[str]         # length n_leads, uppercase
    source: str                   # "wfdb" | "csv"
    notes: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_leads(self) -> int:
        return int(self.signal.shape[1]) if self.signal.ndim == 2 else 1


@dataclass
class IngestReport:
    """Result of validate(): what blocks, what to warn, what stays enabled."""
    hard_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Which downstream feature families are safe to compute on this upload.
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        "single_lead": True,   # rate/rhythm/QRS (needs 1 lead + fs)
        "multilead": False,    # voltage/territorial (needs 12 leads, right order, mV)
        "embedding": False,    # learned model (needs 12 leads, right order; unit-safe)
        "voltage": False,      # mV-amplitude criteria specifically (unit-sensitive)
    })

    @property
    def ok(self) -> bool:
        return not self.hard_errors


# ---------------------------------------------------------------------------
# Readers -> canonical IngestedECG
# ---------------------------------------------------------------------------
def load_wfdb(record_path) -> IngestedECG:
    """Read a WFDB record (path WITHOUT extension; .hea + .dat must sit together).

    wfdb.rdsamp returns the signal already in physical units (mV) using the
    header's gain/baseline, and the lead names + fs from the header - so the
    clean case needs no user-supplied unit/fs/lead-order at all.
    """
    import wfdb
    signal, fields = wfdb.rdsamp(str(record_path))
    lead_names = [str(n).upper() for n in fields.get("sig_name", [])]
    fs = float(fields.get("fs"))
    units = [str(u).lower() for u in fields.get("units", [])]
    notes = [f"WFDB header: fs={fs:g} Hz, {signal.shape[1]} leads, units={set(units) or 'n/a'}"]
    sig = np.asarray(signal, dtype=float)
    # Header units are usually already mV; convert if the header says microvolts.
    if units and all(u in ("uv", "µv", "microvolt", "microvolts") for u in units):
        sig = sig / 1000.0
        notes.append("Converted uV -> mV from header units.")
    sig, lead_names, reordered = reorder_to_ptbxl(sig, lead_names)
    if reordered:
        notes.append("Reordered leads to PTB-XL order from header names.")
    return IngestedECG(signal=sig, fs=fs, lead_names=lead_names, source="wfdb", notes=notes)


def load_csv(path_or_buffer, fs, unit="mV", orientation="auto",
             leads_in_ptbxl_order=True) -> IngestedECG:
    """Read a CSV of raw samples into the canonical form.

    Inputs:
        path_or_buffer: CSV path or file-like.
        fs: sampling frequency (Hz) - REQUIRED, a CSV carries no header for it.
        unit: "mV" or "uV" (uV is divided by 1000). "raw"/ADC is NOT convertible
            without a gain, so it is rejected here (the caller should warn).
        orientation: "auto" (longer axis = samples), "samples_rows" (rows are
            samples, columns leads), or "leads_rows" (transpose of that).
        leads_in_ptbxl_order: if the CSV has 12 columns and the caller confirms
            they are in PTB-XL order, lead names are assigned accordingly;
            otherwise leads are named LEAD1..LEADn (order unconfirmed).
    """
    import pandas as pd
    df = pd.read_csv(path_or_buffer, header=None)
    # Drop a header row if the first row is non-numeric.
    if df.iloc[0].map(lambda v: _is_number(v)).mean() < 0.5:
        df = pd.read_csv(path_or_buffer, header=0) if hasattr(path_or_buffer, "seek") else df.iloc[1:]
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr).all(axis=1)]  # drop all-NaN rows

    if orientation == "leads_rows" or (orientation == "auto" and arr.shape[0] < arr.shape[1]):
        arr = arr.T  # make it (n_samples, n_leads)

    notes = [f"CSV: shape {arr.shape} (samples x leads), fs={fs:g} Hz, unit={unit}"]
    sig = _to_mV(arr, unit, notes)

    n_leads = sig.shape[1]
    if n_leads == 12 and leads_in_ptbxl_order:
        lead_names = list(PTBXL_LEAD_ORDER)
        notes.append("Assumed 12 columns are in PTB-XL lead order (caller-confirmed).")
    else:
        lead_names = [f"LEAD{i+1}" for i in range(n_leads)]
        notes.append("Lead order NOT confirmed - named generically.")
    return IngestedECG(signal=sig, fs=float(fs), lead_names=lead_names, source="csv", notes=notes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _to_mV(signal, unit, notes) -> np.ndarray:
    u = str(unit).strip().lower()
    if u in ("mv", "millivolt", "millivolts"):
        return signal
    if u in ("uv", "µv", "microvolt", "microvolts"):
        notes.append("Converted uV -> mV (/1000).")
        return signal / 1000.0
    raise ValueError(f"Unsupported unit '{unit}' - use mV or uV (raw/ADC needs a gain).")


def reorder_to_ptbxl(signal, lead_names):
    """If lead_names are the 12 standard leads in any order, reorder columns to
    PTB-XL order. Returns (signal, lead_names, reordered_bool)."""
    upper = [str(n).upper() for n in lead_names]
    if len(upper) == 12 and set(upper) == _PTBXL_SET:
        idx = [upper.index(name) for name in PTBXL_LEAD_ORDER]
        return signal[:, idx], list(PTBXL_LEAD_ORDER), (upper != PTBXL_LEAD_ORDER)
    return signal, upper, False


# ---------------------------------------------------------------------------
# Guardrail validation
# ---------------------------------------------------------------------------
def validate(ingested: IngestedECG) -> IngestReport:
    """Inspect an IngestedECG and decide what blocks, what to warn about, and
    which downstream feature families stay enabled. See module docstring."""
    rep = IngestReport()
    sig, fs = ingested.signal, ingested.fs

    # --- Hard errors (block the run) ---
    if sig.ndim != 2 or sig.size == 0:
        rep.hard_errors.append("Empty or malformed signal (expected a 2-D samples x leads array).")
        return rep
    if not np.isfinite(fs) or fs <= 0:
        rep.hard_errors.append("Sampling frequency (fs) is missing or invalid - provide a positive value.")
    if np.isnan(sig).all():
        rep.hard_errors.append("Signal is entirely NaN.")
    if sig.shape[0] < 2:
        rep.hard_errors.append("Signal too short to analyze.")
    if rep.hard_errors:
        return rep

    n_leads = sig.shape[1]
    names_upper = [n.upper() for n in ingested.lead_names]
    has_12 = n_leads == 12
    order_ok = has_12 and names_upper == PTBXL_LEAD_ORDER

    # --- Amplitude / unit plausibility (soft) ---
    p99 = float(np.nanpercentile(np.abs(sig), 99))
    unit_plausible = _MV_PLAUSIBLE_LOW <= p99 <= _MV_PLAUSIBLE_HIGH
    if not unit_plausible:
        if p99 > _MV_PLAUSIBLE_HIGH:
            rep.warnings.append(
                f"Amplitudes look ~{p99:.0f} mV (99th pct) - implausibly large: the unit is "
                "probably µV or raw ADC, not mV. Voltage-amplitude criteria (Sokolow-Lyon "
                "LVH, territorial Q flags) may be UNRELIABLE here - verify the unit. "
                "(The learned embedding is unaffected: it re-normalizes per lead.)")
        else:
            rep.warnings.append(
                f"Amplitudes look very small (99th pct {p99:.3f} mV) - voltage-amplitude "
                "criteria may be UNRELIABLE; check the unit/scaling. (The embedding is unaffected.)")

    # --- Lead count / order (soft, but gates multilead + embedding) ---
    if not has_12:
        rep.warnings.append(
            f"Only {n_leads} lead(s) provided - the learned embedding and multi-lead "
            "voltage features need all 12 leads and are DISABLED. Rate/rhythm/QRS "
            "(single-lead) still run.")
    elif not order_ok:
        rep.warnings.append(
            "12 leads present but their order could not be confirmed as PTB-XL "
            "(I, II, III, aVR, aVL, aVF, V1-V6). The embedding is order-sensitive and is "
            "DISABLED; enable it only if you are sure of the order.")

    # --- Compose the enabled map ---
    # multilead and embedding share the SAME gate - both flow through
    # compute_feature_dict's single multilead_signal argument, and both need all
    # 12 leads in PTB-XL order. The embedding is unit-invariant (per-lead
    # z-score), so the unit does NOT gate them; it only flags whether the
    # voltage-amplitude criteria among the multilead features are trustworthy.
    rep.enabled["single_lead"] = True
    rep.enabled["multilead"] = has_12 and order_ok
    rep.enabled["embedding"] = has_12 and order_ok
    rep.enabled["voltage"] = has_12 and order_ok and unit_plausible
    return rep
