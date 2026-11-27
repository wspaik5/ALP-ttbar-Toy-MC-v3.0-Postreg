#!/usr/bin/env python3
"""v3.0 Post-Registration: Δχ²_z Profiling of Removed Near-Band Vertices.

Implements four investigations preregistered in DOI 10.5281/zenodo.24554495:
  A) Δχ²_z distribution comparison (removed vs surviving)
  B) λ–Δχ²_z correlation and mechanism
  C) Per-band decomposition
  D) λ decomposition (λ_pos, λ_fit)

Configuration: E3 (f = 0.25, η = 2.0, 17-band geometry)
Seeds: i = 50–99 (50 seeds)
Focus mass: 17 MeV

Frozen modules: endcap_sim.py, mtag_v24.py, mtag_v22.py
(MD5 verified before execution)
"""

import os, sys, json, csv, time, hashlib
import numpy as np
from scipy.stats import spearmanr

# ── MD5 verification of frozen modules ──
_FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frozen')
EXPECTED_MD5 = {
    'endcap_sim.py': '7daa81b2723a1c13df68ee69b428c2d0',
    'mtag_v24.py':   '39451a2189d71ec4a9d5990eacb54c6e',
    'mtag_v22.py':   'a687701efe4476a999e27b32341c8c39',
}

print("=" * 70)
print("MD5 Verification of Frozen Modules")
print("=" * 70)
for fname, expected in EXPECTED_MD5.items():
    path = os.path.join(_FROZEN, fname)
    actual = hashlib.md5(open(path, 'rb').read()).hexdigest()
    status = '✓' if actual == expected else '✗ MISMATCH'
    print(f"  {fname}: {actual}  {status}")
    if actual != expected:
        print(f"    Expected: {expected}")
        sys.exit(f"FATAL: MD5 mismatch for {fname}. Aborting.")
print("All frozen modules verified.\n")

# ── Set material scale BEFORE importing endcap_sim (E3: f = 0.25) ──
os.environ['V16_F'] = '0.25'

sys.path.insert(0, _FROZEN)
import endcap_sim
import mtag_v24
import mtag_v22

# ════════════════════════════════════════════════════════════
# Parameters (from preregistration DOI 10.5281/zenodo.24554495)
# ════════════════════════════════════════════════════════════
M_ALP       = 0.017        # 17 MeV in GeV
N_SIG_CAP   = 50_000
N_CONV_CAP  = 100_000
K_TRIM      = 0.01         # 1% from each tail → 2% total
SEEDS       = list(range(50, 100))       # 50 seeds

# Band geometry from 17-band tag
BAND_Z   = np.array(sorted([d[0] for d in mtag_v24.DISKS]))
BAND_HW  = np.array([d[2] for d in sorted(mtag_v24.DISKS, key=lambda d: d[0])])
BAND_X0  = np.array([d[1] for d in sorted(mtag_v24.DISKS, key=lambda d: d[0])])
_gaps    = np.diff(BAND_Z)
D_BAND   = float(_gaps.min() / 2.0)   # 17.5 mm

TGRID = mtag_v24.TGRID

# Which of the 17 bands are sensor disks vs service bands
_sensor_z  = set(d[0] for d in mtag_v24._SENSOR_DISKS)
IS_SENSOR  = np.array([z in _sensor_z for z in BAND_Z])

_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════

def scan_with_window(sig, bkg, lam_sig, lam_conv):
    """Scan tag threshold × mass window; return best configuration."""
    m = M_ALP
    S0 = sig['w'][sig['base']].sum()
    lim0 = endcap_sim.upper_limit(S0, 0)
    best = None

    cands = [(np.ones(len(sig['z']), bool),
              np.ones(len(bkg['z']), bool), None, None)]
    for lo in np.linspace(0, 1.5 * m, 31):
        for hi in (3 * m + 0.01, np.inf):
            cands.append(((sig['m'] > lo) & (sig['m'] < hi),
                          (bkg['m'] > lo) & (bkg['m'] < hi), lo, hi))

    for t in TGRID:
        if t is None:
            ts = np.ones(len(sig['z']), bool)
            tb = np.ones(len(bkg['z']), bool)
        else:
            ts = lam_sig >= t
            tb = lam_conv >= t
        for ms, mb, lo, hi in cands:
            cs = sig['base'] & ts & ms
            cb = bkg['base'] & tb & mb
            S = sig['w'][cs].sum()
            B = bkg['w'][cb].sum()
            lim = endcap_sim.upper_limit(S, B)
            if best is None or lim < best['limit']:
                best = dict(limit=lim, deg=lim / lim0, t=t,
                            B=float(B), S=float(S), m_lo=lo, m_hi=hi)
    return best


def apply_trim(lam_vals, base_mask, k=K_TRIM):
    """Apply k% trim from each tail of the λ distribution.

    Tie-breaking: ascending event index (preregistered rule).
    Returns (removal_mask, n_rem_per_tail, removed_idx, lower_idx, upper_idx).
    """
    base_indices = np.where(base_mask)[0]
    n_base = len(base_indices)
    n_rem = int(np.floor(k * n_base))

    if n_rem == 0:
        empty = np.array([], dtype=int)
        return np.zeros(len(lam_vals), bool), 0, empty, empty, empty

    lam_base = lam_vals[base_indices]
    # Sort by λ value, then by original index (ascending) for tie-breaking
    order = np.argsort(lam_base, kind='stable')  # stable sort preserves index order for ties
    lower_local = order[:n_rem]
    upper_local = order[-n_rem:]
    removed_local = np.union1d(lower_local, upper_local)

    removed_full = base_indices[removed_local]
    lower_full  = base_indices[lower_local]
    upper_full  = base_indices[upper_local]

    mask = np.zeros(len(lam_vals), bool)
    mask[removed_full] = True
    return mask, n_rem, removed_full, lower_full, upper_full


def classify_z(zm):
    """Near-band / inter-band classification based on d_band threshold."""
    dist = np.abs(zm[:, None] - BAND_Z[None, :]).min(axis=1)
    near_band = dist <= D_BAND
    return near_band


def nearest_band_idx(zm):
    """Return index of nearest band for each vertex."""
    return np.argmin(np.abs(zm[:, None] - BAND_Z[None, :]), axis=1)


def compute_dchi2_z(zm, sz):
    """Compute Δχ²_z = (Δz/σ_z)² where Δz = z_m − Z_nearest."""
    dists = np.abs(zm[:, None] - BAND_Z[None, :])
    min_dist = dists.min(axis=1)  # |Δz| = distance to nearest band centre
    return (min_dist / sz) ** 2


def compute_lam_pos(zm, sz):
    """Position-only tag: λ_pos = ln p_sig(z) − ln max(p_conv(z, σ_z), P_FLOOR).
    i.e. full λ with Δχ²_3D/2 set to zero."""
    return np.log(mtag_v24.p_sig(zm)) - np.log(np.maximum(mtag_v24.p_conv(zm, sz), mtag_v24.P_FLOOR))


def compute_lam_fit(dchi2_3D):
    """Fit-only tag: λ_fit = Δχ²_3D/2.
    i.e. full λ with position log-likelihood ratio set to zero."""
    d = np.where(np.isfinite(dchi2_3D), dchi2_3D, 0.0)
    return 0.5 * d


def nearest_band_pconv_fraction(zm, sz):
    """For each vertex, compute fraction of p_conv from the nearest band alone."""
    Z = np.array([d[0] for d in mtag_v24.DISKS], float)
    X = np.array([d[1] for d in mtag_v24.DISKS], float)
    H = np.array([d[2] for d in mtag_v24.DISKS], float)
    c = X / X.sum()

    from scipy.special import ndtr

    zm_2d = np.asarray(zm, float)[:, None]
    sz_2d = np.maximum(np.asarray(sz, float), 1e-6)[:, None]

    # Per-band contribution
    contrib = c * (ndtr((zm_2d - Z + H) / sz_2d) - ndtr((zm_2d - Z - H) / sz_2d)) / (2 * H)
    total = contrib.sum(axis=1)
    nearest_idx = np.argmin(np.abs(zm_2d - Z), axis=1)
    nearest_contrib = contrib[np.arange(len(zm)), nearest_idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(total > 0, nearest_contrib / total, 0.0)
    return frac


# ════════════════════════════════════════════════════════════
# Per-seed runner: all four investigations
# ════════════════════════════════════════════════════════════

def run_seed(i, save_rep=False):
    """Run one seed for Investigations A, B, C, D.  Returns result dict."""
    t0 = time.time()
    seed_conv = 100 * i + 5
    seed_sig  = 100 * i + 11

    # Generate
    conv = endcap_sim.endcap_conversions(n=1_500_000, seed=seed_conv, config='E2')
    sig  = endcap_sim.endcap_signal(M_ALP, n=300_000, seed=seed_sig, config='E2')

    # Subsample
    conv_s, _ = endcap_sim.subsample(conv, N_CONV_CAP)
    sig_s,  _ = endcap_sim.subsample(sig,  N_SIG_CAP)

    # Δχ²_3D (from frozen endcap_sim)
    dchi2_c, _ = endcap_sim.compute_dchi2_endcap(conv_s)
    dchi2_s, _ = endcap_sim.compute_dchi2_endcap(sig_s)

    # Full λ (17-band tag)
    lam_c = mtag_v24.lam(conv_s['zm'], conv_s['sz'], dchi2_c)
    lam_s = mtag_v24.lam(sig_s['zm'], sig_s['sz'], dchi2_s)

    n_base = int(conv_s['base'].sum())
    base_idx = np.where(conv_s['base'])[0]

    # ── Trimming (full λ) ──
    mask, n_rem, removed, lower, upper = apply_trim(lam_c, conv_s['base'])
    n_removed = int(mask.sum())

    # ── Near-band classification ──
    near_all = classify_z(conv_s['zm'][base_idx])
    near_rem = classify_z(conv_s['zm'][removed])

    # Δχ²_z for all base and removed
    dchi2z_all = compute_dchi2_z(conv_s['zm'][base_idx], conv_s['sz'][base_idx])
    dchi2z_rem = compute_dchi2_z(conv_s['zm'][removed], conv_s['sz'][removed])

    # Surviving = base minus removed, near-band only
    surviving_mask_in_base = np.ones(len(base_idx), bool)
    for r in removed:
        pos = np.searchsorted(base_idx, r)
        if pos < len(base_idx) and base_idx[pos] == r:
            surviving_mask_in_base[pos] = False

    surv_near = near_all & surviving_mask_in_base
    rem_near = near_rem  # already among removed

    dchi2z_surv_near = dchi2z_all[surv_near]
    dchi2z_rem_near  = dchi2z_rem[rem_near]

    # ════════════════════════════════════════
    # Investigation A: Δχ²_z distribution
    # ════════════════════════════════════════

    # A1: mean ratio
    mean_rem = float(dchi2z_rem_near.mean()) if len(dchi2z_rem_near) > 0 else np.nan
    mean_surv = float(dchi2z_surv_near.mean()) if len(dchi2z_surv_near) > 0 else np.nan
    if np.isnan(mean_rem) or np.isnan(mean_surv) or mean_surv == 0:
        a1_ratio = np.nan
    else:
        a1_ratio = mean_rem / mean_surv

    # A2: enrichment at Δχ²_z > 4
    frac_rem_above4 = float((dchi2z_rem_near > 4).mean()) if len(dchi2z_rem_near) > 0 else np.nan
    frac_surv_above4 = float((dchi2z_surv_near > 4).mean()) if len(dchi2z_surv_near) > 0 else np.nan
    if np.isnan(frac_rem_above4) or np.isnan(frac_surv_above4) or frac_surv_above4 == 0:
        a2_enrichment = np.nan
    else:
        a2_enrichment = frac_rem_above4 / frac_surv_above4

    # A3: (computed across seeds later, but store per-seed mean)
    a3_mean_dchi2z_rem = mean_rem

    # ════════════════════════════════════════
    # Investigation B: λ–Δχ²_z correlation
    # ════════════════════════════════════════

    # B1: Spearman ρ(|λ|, Δχ²_z) among all near-band vertices
    near_base = near_all
    lam_near_base = np.abs(lam_c[base_idx[near_base]])
    dchi2z_near_base = dchi2z_all[near_base]

    if len(lam_near_base) > 2:
        b1_rho, b1_pval = spearmanr(lam_near_base, dchi2z_near_base)
    else:
        b1_rho, b1_pval = np.nan, np.nan

    # B2: upper/lower tail symmetry in Δχ²_z
    in_upper = np.isin(removed, upper)
    in_lower = np.isin(removed, lower)
    dchi2z_upper = compute_dchi2_z(conv_s['zm'][removed[in_upper]], conv_s['sz'][removed[in_upper]])
    dchi2z_lower = compute_dchi2_z(conv_s['zm'][removed[in_lower]], conv_s['sz'][removed[in_lower]])
    # Filter to near-band for upper/lower
    near_upper = classify_z(conv_s['zm'][removed[in_upper]])
    near_lower = classify_z(conv_s['zm'][removed[in_lower]])
    dchi2z_upper_near = dchi2z_upper[near_upper]
    dchi2z_lower_near = dchi2z_lower[near_lower]

    mean_up = float(dchi2z_upper_near.mean()) if len(dchi2z_upper_near) > 0 else np.nan
    mean_lo = float(dchi2z_lower_near.mean()) if len(dchi2z_lower_near) > 0 else np.nan
    if np.isnan(mean_up) or np.isnan(mean_lo) or (mean_up + mean_lo) == 0:
        b2_asymmetry = np.nan
    else:
        pooled = (mean_up + mean_lo) / 2.0
        b2_asymmetry = abs(mean_up - mean_lo) / pooled

    # B3: AUC using logistic regression with Δχ²_z to predict removal
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Labels: 1 = removed near-band, 0 = surviving near-band
    n_rem_near = int(rem_near.sum())
    n_surv_near = int(surv_near.sum())
    if n_rem_near > 0 and n_surv_near > 0:
        X_lr = np.concatenate([dchi2z_rem_near, dchi2z_surv_near]).reshape(-1, 1)
        y_lr = np.concatenate([np.ones(n_rem_near), np.zeros(n_surv_near)])
        try:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_lr, y_lr)
            prob = lr.predict_proba(X_lr)[:, 1]
            b3_auc = float(roc_auc_score(y_lr, prob))
        except Exception:
            b3_auc = np.nan
    else:
        b3_auc = np.nan

    # ════════════════════════════════════════
    # Investigation C: Per-band decomposition
    # ════════════════════════════════════════

    # Per-band removal rates
    band_idx_all = nearest_band_idx(conv_s['zm'][base_idx])
    band_idx_rem = nearest_band_idx(conv_s['zm'][removed])
    near_rem_mask = near_rem

    per_band_rates = []
    for b in range(17):
        n_near_b = int(((band_idx_all == b) & near_all).sum())
        n_rem_b  = int(((band_idx_rem == b) & near_rem_mask).sum())
        if n_near_b > 0:
            per_band_rates.append(float(n_rem_b / n_near_b))
        else:
            per_band_rates.append(np.nan)

    evaluable = [r for r in per_band_rates if not np.isnan(r)]
    c1_cv = float(np.std(evaluable) / np.mean(evaluable)) if len(evaluable) > 1 and np.mean(evaluable) > 0 else np.nan

    # C2: Spearman ρ(removal_rate, band_HW)
    eval_mask = [not np.isnan(r) for r in per_band_rates]
    eval_rates = np.array([r for r, m in zip(per_band_rates, eval_mask) if m])
    eval_hw = BAND_HW[eval_mask]
    if len(eval_rates) > 2:
        c2_rho, c2_pval = spearmanr(eval_rates, eval_hw)
    else:
        c2_rho, c2_pval = np.nan, np.nan

    # C3: nearest-band p_conv fraction among removed near-band
    if n_rem_near > 0:
        zm_rem_nb = conv_s['zm'][removed[rem_near]]
        sz_rem_nb = conv_s['sz'][removed[rem_near]]
        nb_frac = nearest_band_pconv_fraction(zm_rem_nb, sz_rem_nb)
        c3_mean_frac = float(nb_frac.mean())
    else:
        c3_mean_frac = np.nan

    # ════════════════════════════════════════
    # Investigation D: λ decomposition
    # ════════════════════════════════════════

    # λ_pos (position-only) — each decomposed tag uses its OWN surviving mean
    lam_pos_c = compute_lam_pos(conv_s['zm'], conv_s['sz'])
    mask_pos, _, removed_pos, lower_pos, upper_pos = apply_trim(lam_pos_c, conv_s['base'])
    near_rem_pos = classify_z(conv_s['zm'][removed_pos])
    dchi2z_rem_pos = compute_dchi2_z(conv_s['zm'][removed_pos], conv_s['sz'][removed_pos])
    dchi2z_rem_pos_near = dchi2z_rem_pos[near_rem_pos]
    mean_rem_pos = float(dchi2z_rem_pos_near.mean()) if len(dchi2z_rem_pos_near) > 0 else np.nan
    # Surviving under λ_pos trimming
    surv_mask_pos = np.ones(len(base_idx), bool)
    for r in removed_pos:
        pos = np.searchsorted(base_idx, r)
        if pos < len(base_idx) and base_idx[pos] == r:
            surv_mask_pos[pos] = False
    surv_near_pos = near_all & surv_mask_pos
    dchi2z_surv_pos_near = dchi2z_all[surv_near_pos]
    mean_surv_pos = float(dchi2z_surv_pos_near.mean()) if len(dchi2z_surv_pos_near) > 0 else np.nan
    if np.isnan(mean_rem_pos) or np.isnan(mean_surv_pos) or mean_surv_pos == 0:
        d1_ratio = np.nan
    else:
        d1_ratio = mean_rem_pos / mean_surv_pos

    # λ_fit (fit-only) — each decomposed tag uses its OWN surviving mean
    lam_fit_c = compute_lam_fit(dchi2_c)
    mask_fit, _, removed_fit, lower_fit, upper_fit = apply_trim(lam_fit_c, conv_s['base'])
    near_rem_fit = classify_z(conv_s['zm'][removed_fit])
    dchi2z_rem_fit = compute_dchi2_z(conv_s['zm'][removed_fit], conv_s['sz'][removed_fit])
    dchi2z_rem_fit_near = dchi2z_rem_fit[near_rem_fit]
    mean_rem_fit = float(dchi2z_rem_fit_near.mean()) if len(dchi2z_rem_fit_near) > 0 else np.nan
    # Surviving under λ_fit trimming
    surv_mask_fit = np.ones(len(base_idx), bool)
    for r in removed_fit:
        pos = np.searchsorted(base_idx, r)
        if pos < len(base_idx) and base_idx[pos] == r:
            surv_mask_fit[pos] = False
    surv_near_fit = near_all & surv_mask_fit
    dchi2z_surv_fit_near = dchi2z_all[surv_near_fit]
    mean_surv_fit = float(dchi2z_surv_fit_near.mean()) if len(dchi2z_surv_fit_near) > 0 else np.nan
    if np.isnan(mean_rem_fit) or np.isnan(mean_surv_fit) or mean_surv_fit == 0:
        d2_ratio = np.nan
    else:
        d2_ratio = mean_rem_fit / mean_surv_fit

    # Overlap metrics (per-tail)
    def overlap_fraction(set_a, set_b):
        """Fraction of set_a that is also in set_b."""
        if len(set_a) == 0:
            return np.nan
        return float(np.isin(set_a, set_b).sum() / len(set_a))

    overlap_pos_upper = overlap_fraction(removed_pos[np.isin(removed_pos, upper_pos)], upper)
    overlap_pos_lower = overlap_fraction(removed_pos[np.isin(removed_pos, lower_pos)], lower)
    # Actually: overlap of λ_pos removal list with full λ removal list, per tail
    # For upper tail of λ_pos: fraction that are also in upper tail of full λ
    up_pos = set(upper_pos.tolist())
    lo_pos = set(lower_pos.tolist())
    up_full = set(upper.tolist())
    lo_full = set(lower.tolist())
    overlap_pos_upper = float(len(up_pos & up_full) / len(up_pos)) if len(up_pos) > 0 else np.nan
    overlap_pos_lower = float(len(lo_pos & lo_full) / len(lo_pos)) if len(lo_pos) > 0 else np.nan

    up_fit = set(upper_fit.tolist())
    lo_fit = set(lower_fit.tolist())
    overlap_fit_upper = float(len(up_fit & up_full) / len(up_fit)) if len(up_fit) > 0 else np.nan
    overlap_fit_lower = float(len(lo_fit & lo_full) / len(lo_fit)) if len(lo_fit) > 0 else np.nan

    dt = time.time() - t0

    result = dict(
        seed_i          = i,
        time_s          = round(dt, 1),
        n_base_conv     = n_base,
        n_removed       = n_removed,
        n_rem_per_tail  = int(n_rem),
        n_near_rem      = int(rem_near.sum()),
        n_surv_near     = n_surv_near,
        # Investigation A
        a1_mean_rem     = round(mean_rem, 4) if not np.isnan(mean_rem) else None,
        a1_mean_surv    = round(mean_surv, 4) if not np.isnan(mean_surv) else None,
        a1_ratio        = round(a1_ratio, 4) if not np.isnan(a1_ratio) else None,
        a2_frac_rem     = round(frac_rem_above4, 4) if not np.isnan(frac_rem_above4) else None,
        a2_frac_surv    = round(frac_surv_above4, 4) if not np.isnan(frac_surv_above4) else None,
        a2_enrichment   = round(a2_enrichment, 4) if not np.isnan(a2_enrichment) else None,
        a3_mean_dchi2z  = round(a3_mean_dchi2z_rem, 4) if not np.isnan(a3_mean_dchi2z_rem) else None,
        # Investigation B
        b1_rho          = round(b1_rho, 4) if not np.isnan(b1_rho) else None,
        b1_pval         = float(b1_pval) if not np.isnan(b1_pval) else None,
        b2_mean_upper   = round(mean_up, 4) if not np.isnan(mean_up) else None,
        b2_mean_lower   = round(mean_lo, 4) if not np.isnan(mean_lo) else None,
        b2_asymmetry    = round(b2_asymmetry, 4) if not np.isnan(b2_asymmetry) else None,
        b3_auc          = round(b3_auc, 4) if not np.isnan(b3_auc) else None,
        # Investigation C
        c1_per_band_rates = [round(r, 6) if not np.isnan(r) else None for r in per_band_rates],
        c1_cv           = round(c1_cv, 4) if not np.isnan(c1_cv) else None,
        c2_rho          = round(c2_rho, 4) if not np.isnan(c2_rho) else None,
        c2_pval         = float(c2_pval) if not np.isnan(c2_pval) else None,
        c3_mean_frac    = round(c3_mean_frac, 4) if not np.isnan(c3_mean_frac) else None,
        # Investigation D
        d1_mean_rem_pos  = round(mean_rem_pos, 4) if not np.isnan(mean_rem_pos) else None,
        d1_mean_surv_pos = round(mean_surv_pos, 4) if not np.isnan(mean_surv_pos) else None,
        d1_ratio         = round(d1_ratio, 4) if not np.isnan(d1_ratio) else None,
        d2_mean_rem_fit  = round(mean_rem_fit, 4) if not np.isnan(mean_rem_fit) else None,
        d2_mean_surv_fit = round(mean_surv_fit, 4) if not np.isnan(mean_surv_fit) else None,
        d2_ratio         = round(d2_ratio, 4) if not np.isnan(d2_ratio) else None,
        overlap_pos_upper = round(overlap_pos_upper, 4) if not np.isnan(overlap_pos_upper) else None,
        overlap_pos_lower = round(overlap_pos_lower, 4) if not np.isnan(overlap_pos_lower) else None,
        overlap_fit_upper = round(overlap_fit_upper, 4) if not np.isnan(overlap_fit_upper) else None,
        overlap_fit_lower = round(overlap_fit_lower, 4) if not np.isnan(overlap_fit_lower) else None,
    )

    rep_data = None
    if save_rep:
        rep_data = dict(
            zm_all_base    = conv_s['zm'][base_idx].copy(),
            sz_all_base    = conv_s['sz'][base_idx].copy(),
            lam_all_base   = lam_c[base_idx].copy(),
            dchi2z_all     = dchi2z_all.copy(),
            near_all       = near_all.copy(),
            zm_removed     = conv_s['zm'][removed].copy(),
            sz_removed     = conv_s['sz'][removed].copy(),
            lam_removed    = lam_c[removed].copy(),
            dchi2z_removed = dchi2z_rem.copy(),
            near_removed   = near_rem.copy(),
            surviving_mask = surviving_mask_in_base.copy(),
            in_upper       = in_upper.copy(),
            in_lower       = in_lower.copy(),
            # D decomposition
            lam_pos_all    = lam_pos_c[base_idx].copy(),
            lam_fit_all    = lam_fit_c[base_idx].copy(),
        )

    return result, rep_data


# ════════════════════════════════════════════════════════════
# Prediction evaluation
# ════════════════════════════════════════════════════════════

def evaluate_predictions(results):
    """Evaluate 11 preregistered predictions.  Returns list of dicts."""

    def safe_arr(key):
        vals = [r[key] for r in results if r[key] is not None]
        return np.array(vals) if vals else np.array([])

    # A1
    a1_arr = safe_arr('a1_ratio')
    n_a1_na = sum(1 for r in results if r['a1_ratio'] is None)
    a1_mean = float(a1_arr.mean()) if len(a1_arr) > 0 else np.nan

    # A2
    a2_arr = safe_arr('a2_enrichment')
    n_a2_na = sum(1 for r in results if r['a2_enrichment'] is None)
    a2_mean = float(a2_arr.mean()) if len(a2_arr) > 0 else np.nan

    # A3
    a3_arr = safe_arr('a3_mean_dchi2z')
    a3_cv = float(a3_arr.std() / a3_arr.mean()) if len(a3_arr) > 0 and a3_arr.mean() > 0 else np.nan

    # B1
    b1_arr = safe_arr('b1_rho')
    b1_mean = float(b1_arr.mean()) if len(b1_arr) > 0 else np.nan

    # B2
    b2_arr = safe_arr('b2_asymmetry')
    b2_mean = float(b2_arr.mean()) if len(b2_arr) > 0 else np.nan

    # B3
    b3_arr = safe_arr('b3_auc')
    b3_mean = float(b3_arr.mean()) if len(b3_arr) > 0 else np.nan

    # C1
    c1_arr = safe_arr('c1_cv')
    c1_mean = float(c1_arr.mean()) if len(c1_arr) > 0 else np.nan

    # C2
    c2_arr = safe_arr('c2_rho')
    c2_mean = float(c2_arr.mean()) if len(c2_arr) > 0 else np.nan

    # C3
    c3_arr = safe_arr('c3_mean_frac')
    c3_mean = float(c3_arr.mean()) if len(c3_arr) > 0 else np.nan

    # D1
    d1_arr = safe_arr('d1_ratio')
    n_d1_na = sum(1 for r in results if r['d1_ratio'] is None)
    d1_mean = float(d1_arr.mean()) if len(d1_arr) > 0 else np.nan

    # D2
    d2_arr = safe_arr('d2_ratio')
    n_d2_na = sum(1 for r in results if r['d2_ratio'] is None)
    d2_mean = float(d2_arr.mean()) if len(d2_arr) > 0 else np.nan

    # Overlap metrics
    op_u = safe_arr('overlap_pos_upper')
    op_l = safe_arr('overlap_pos_lower')
    of_u = safe_arr('overlap_fit_upper')
    of_l = safe_arr('overlap_fit_lower')

    preds = [
        dict(id='A1', desc='Removed have higher mean Δχ²_z',
             criterion='mean ratio > 2.0', tier='Core',
             value=round(a1_mean, 4) if not np.isnan(a1_mean) else None,
             detail=f'Mean ratio = {a1_mean:.4f} (N/A: {n_a1_na} seeds)' if not np.isnan(a1_mean) else 'Not evaluable',
             std=round(float(a1_arr.std()), 4) if len(a1_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(a1_mean) and a1_mean > 2.0 else ('NOT EVALUABLE' if np.isnan(a1_mean) else 'REFUTED')),

        dict(id='A2', desc='Removed enriched at Δχ²_z > 4',
             criterion='enrichment > 3×', tier='Core',
             value=round(a2_mean, 4) if not np.isnan(a2_mean) else None,
             detail=f'Mean enrichment = {a2_mean:.2f}× (N/A: {n_a2_na} seeds)' if not np.isnan(a2_mean) else 'Not evaluable',
             std=round(float(a2_arr.std()), 4) if len(a2_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(a2_mean) and a2_mean > 3.0 else ('NOT EVALUABLE' if np.isnan(a2_mean) else 'REFUTED')),

        dict(id='A3', desc='Δχ²_z of removed consistent across seeds',
             criterion='CV < 50%', tier='Secondary',
             value=round(a3_cv * 100, 2) if not np.isnan(a3_cv) else None,
             detail=f'CV = {a3_cv*100:.2f}%' if not np.isnan(a3_cv) else 'Not evaluable',
             result='CONFIRMED' if not np.isnan(a3_cv) and a3_cv < 0.50 else ('NOT EVALUABLE' if np.isnan(a3_cv) else 'REFUTED')),

        dict(id='B1', desc='|λ| correlates with Δχ²_z',
             criterion='Spearman ρ > 0.3', tier='Core',
             value=round(b1_mean, 4) if not np.isnan(b1_mean) else None,
             detail=f'Mean ρ = {b1_mean:.4f}' if not np.isnan(b1_mean) else 'Not evaluable',
             std=round(float(b1_arr.std()), 4) if len(b1_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(b1_mean) and b1_mean > 0.3 else ('NOT EVALUABLE' if np.isnan(b1_mean) else 'REFUTED')),

        dict(id='B2', desc='Upper/lower tails symmetric in Δχ²_z',
             criterion='|Δmean|/pooled < 0.2', tier='Diagnostic',
             value=round(b2_mean, 4) if not np.isnan(b2_mean) else None,
             detail=f'Mean asymmetry = {b2_mean:.4f}' if not np.isnan(b2_mean) else 'Not evaluable',
             std=round(float(b2_arr.std()), 4) if len(b2_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(b2_mean) and b2_mean < 0.2 else ('NOT EVALUABLE' if np.isnan(b2_mean) else 'REFUTED')),

        dict(id='B3', desc='Δχ²_z discriminates removal',
             criterion='AUC > 0.7', tier='Secondary',
             value=round(b3_mean, 4) if not np.isnan(b3_mean) else None,
             detail=f'Mean AUC = {b3_mean:.4f}' if not np.isnan(b3_mean) else 'Not evaluable',
             std=round(float(b3_arr.std()), 4) if len(b3_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(b3_mean) and b3_mean > 0.7 else ('NOT EVALUABLE' if np.isnan(b3_mean) else 'REFUTED')),

        dict(id='C1', desc='Per-band removal rate varies',
             criterion='CV > 20%', tier='Diagnostic',
             value=round(c1_mean * 100, 2) if not np.isnan(c1_mean) else None,
             detail=f'Mean CV = {c1_mean*100:.2f}%' if not np.isnan(c1_mean) else 'Not evaluable',
             std=round(float(c1_arr.std() * 100), 2) if len(c1_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(c1_mean) and c1_mean > 0.20 else ('NOT EVALUABLE' if np.isnan(c1_mean) else 'REFUTED')),

        dict(id='C2', desc='Removal rate anti-correlates with band HW',
             criterion='Spearman ρ < 0', tier='Diagnostic',
             value=round(c2_mean, 4) if not np.isnan(c2_mean) else None,
             detail=f'Mean ρ = {c2_mean:.4f}' if not np.isnan(c2_mean) else 'Not evaluable',
             std=round(float(c2_arr.std()), 4) if len(c2_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(c2_mean) and c2_mean < 0 else ('NOT EVALUABLE' if np.isnan(c2_mean) else 'REFUTED')),

        dict(id='C3', desc='Nearest-band p_conv dominates',
             criterion='fraction > 0.8', tier='Secondary',
             value=round(c3_mean, 4) if not np.isnan(c3_mean) else None,
             detail=f'Mean nearest-band fraction = {c3_mean:.4f}' if not np.isnan(c3_mean) else 'Not evaluable',
             std=round(float(c3_arr.std()), 4) if len(c3_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(c3_mean) and c3_mean > 0.8 else ('NOT EVALUABLE' if np.isnan(c3_mean) else 'REFUTED')),

        dict(id='D1', desc='Position-only λ_pos still shows A1 effect',
             criterion='mean ratio > 2.0', tier='Core',
             value=round(d1_mean, 4) if not np.isnan(d1_mean) else None,
             detail=f'Mean ratio = {d1_mean:.4f} (N/A: {n_d1_na} seeds)' if not np.isnan(d1_mean) else 'Not evaluable',
             std=round(float(d1_arr.std()), 4) if len(d1_arr) > 0 else None,
             result='CONFIRMED' if not np.isnan(d1_mean) and d1_mean > 2.0 else ('NOT EVALUABLE' if np.isnan(d1_mean) else 'REFUTED')),

        dict(id='D2', desc='Fit-only λ_fit channel contribution',
             criterion='mean ratio reported', tier='Diagnostic',
             value=round(d2_mean, 4) if not np.isnan(d2_mean) else None,
             detail=f'Mean ratio = {d2_mean:.4f} (N/A: {n_d2_na} seeds)' if not np.isnan(d2_mean) else 'Not evaluable',
             std=round(float(d2_arr.std()), 4) if len(d2_arr) > 0 else None,
             result='REPORTED'),  # D2 is diagnostic, no pass/fail
    ]

    # Overlap summary
    overlap = dict(
        pos_upper_mean = round(float(op_u.mean()), 4) if len(op_u) > 0 else None,
        pos_upper_std  = round(float(op_u.std()), 4) if len(op_u) > 0 else None,
        pos_lower_mean = round(float(op_l.mean()), 4) if len(op_l) > 0 else None,
        pos_lower_std  = round(float(op_l.std()), 4) if len(op_l) > 0 else None,
        fit_upper_mean = round(float(of_u.mean()), 4) if len(of_u) > 0 else None,
        fit_upper_std  = round(float(of_u.std()), 4) if len(of_u) > 0 else None,
        fit_lower_mean = round(float(of_l.mean()), 4) if len(of_l) > 0 else None,
        fit_lower_std  = round(float(of_l.std()), 4) if len(of_l) > 0 else None,
    )

    return preds, overlap


# ════════════════════════════════════════════════════════════
# CSV export
# ════════════════════════════════════════════════════════════

def save_csv(results, predictions, overlap, dir_path):
    """Save per-seed raw statistics and prediction scorecard as CSV."""

    # 1. Per-seed raw statistics
    csv_path = os.path.join(dir_path, 'v30_postreg_per_seed_statistics.csv')
    fields = [
        'seed_i', 'n_base_conv', 'n_removed', 'n_rem_per_tail', 'n_near_rem', 'n_surv_near',
        'a1_mean_rem', 'a1_mean_surv', 'a1_ratio',
        'a2_frac_rem', 'a2_frac_surv', 'a2_enrichment', 'a3_mean_dchi2z',
        'b1_rho', 'b1_pval', 'b2_mean_upper', 'b2_mean_lower', 'b2_asymmetry', 'b3_auc',
        'c1_cv', 'c2_rho', 'c2_pval', 'c3_mean_frac',
        'd1_mean_rem_pos', 'd1_ratio', 'd2_mean_rem_fit', 'd2_ratio',
        'overlap_pos_upper', 'overlap_pos_lower', 'overlap_fit_upper', 'overlap_fit_lower',
        'time_s',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"  Saved: {csv_path}")

    # 2. Prediction scorecard
    sc_path = os.path.join(dir_path, 'v30_postreg_prediction_scorecard.csv')
    with open(sc_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ID', 'Description', 'Criterion', 'Tier', 'Value', 'Std', 'Detail', 'Result'])
        for p in predictions:
            w.writerow([p['id'], p['desc'], p['criterion'], p['tier'],
                        p.get('value', ''), p.get('std', ''), p['detail'], p['result']])
    print(f"  Saved: {sc_path}")

    return csv_path, sc_path


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("v3.0 Post-Registration: Δχ²_z Profiling of Removed Near-Band Vertices")
    print(f"Config: E3 (f = 0.25), m = 17 MeV, 17-band geometry")
    print(f"d_band = {D_BAND} mm (half minimum inter-band gap)")
    print(f"Seeds: {SEEDS[0]}–{SEEDS[-1]} (50 seeds)")
    print(f"Investigations: A (Δχ²_z distribution), B (λ–Δχ²_z correlation),")
    print(f"                C (per-band decomposition), D (λ decomposition)")
    print("=" * 70)

    # ── Run all seeds ──
    print("\n▶ Running 50-seed ensemble (Investigations A, B, C, D)")
    results = []
    rep_data = None
    t_total = time.time()

    for idx, i in enumerate(SEEDS):
        save_rep = (i == SEEDS[0])
        r, rd = run_seed(i, save_rep=save_rep)
        results.append(r)
        if rd is not None:
            rep_data = rd
        a1_str = f"A1={r['a1_ratio']:.2f}" if r['a1_ratio'] is not None else "A1=N/A"
        b1_str = f"B1={r['b1_rho']:.3f}" if r['b1_rho'] is not None else "B1=N/A"
        d1_str = f"D1={r['d1_ratio']:.2f}" if r['d1_ratio'] is not None else "D1=N/A"
        print(f"  [{idx+1:2d}/50]  seed {i}: {a1_str}  {b1_str}  B3={r['b3_auc']:.3f}  {d1_str}  ({r['time_s']:.1f}s)")

    dt_total = time.time() - t_total
    print(f"  → All 50 seeds done in {dt_total:.0f}s")

    # ── Evaluate predictions ──
    print("\n▶ Evaluating 11 predictions...")
    predictions, overlap = evaluate_predictions(results)

    for p in predictions:
        status = '✓' if p['result'] == 'CONFIRMED' else ('?' if p['result'] in ('NOT EVALUABLE', 'REPORTED') else '✗')
        print(f"  {status} {p['id']} ({p['tier']}): {p['result']}  —  {p['detail']}")

    core = [p for p in predictions if p['tier'] == 'Core']
    n_core = sum(1 for p in core if p['result'] == 'CONFIRMED')
    print(f"\n  Core predictions: {n_core}/4 CONFIRMED")
    strength = 'STRONG' if n_core == 4 else ('PARTIAL' if 2 <= n_core <= 3 else 'WEAK')
    print(f"  Overall: {strength} support for Δχ²_z hypothesis")

    # Overlap summary
    print(f"\n  λ_pos overlap — upper: {overlap['pos_upper_mean']:.1%} ± {overlap['pos_upper_std']:.1%}, "
          f"lower: {overlap['pos_lower_mean']:.1%} ± {overlap['pos_lower_std']:.1%}")
    print(f"  λ_fit overlap — upper: {overlap['fit_upper_mean']:.1%} ± {overlap['fit_upper_std']:.1%}, "
          f"lower: {overlap['fit_lower_mean']:.1%} ± {overlap['fit_lower_std']:.1%}")

    # ── Save JSON ──
    out = dict(
        config = dict(
            m_alp_GeV=M_ALP, f_mat=0.25, eta_ref=2.0,
            n_bands=17, d_band_mm=D_BAND, k_trim=K_TRIM,
            n_seeds=len(SEEDS), seed_range=[SEEDS[0], SEEDS[-1]],
            prereg_doi='10.5281/zenodo.24554495',
            postreg_doi='10.5281/zenodo.24581149',
        ),
        results = results,
        predictions = predictions,
        overlap = overlap,
        summary = dict(
            n_core_confirmed = n_core,
            overall_strength = strength,
            runtime_s = round(dt_total, 1),
        ),
    )

    json_path = os.path.join(_DIR, 'v30_postreg_results.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # ── Save CSV ──
    print("\n▶ Saving CSV files...")
    save_csv(results, predictions, overlap, _DIR)

    # ── Save representative arrays for figures ──
    npz_path = os.path.join(_DIR, 'v30_postreg_data.npz')
    np.savez_compressed(npz_path, **{k: v for k, v in rep_data.items()})
    print(f"  Saved: {npz_path}")

    total = time.time() - t_total
    print(f"\n{'=' * 70}")
    n_confirmed = sum(1 for p in predictions if p['result'] == 'CONFIRMED')
    print(f"All done in {total:.0f}s.  {n_confirmed}/11 predictions evaluated.")
    print(f"Core: {n_core}/4 → {strength} support")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
