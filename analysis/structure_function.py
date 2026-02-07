#!/usr/bin/env python3
"""
Structure Function (SF) minimum-variability timescale (MVT) estimator
for Poisson-distributed binned counts with per-bin background (counts/bin)
and a single uniform dt.

Inputs:
  - counts: observed counts/bin (source + background)
  - bg_counts: estimated background counts/bin (may vary with time)
  - dt: bin width (seconds)

Computes SF on net rate:
  x_i = (counts_i - bg_counts_i) / dt

Analytic Poisson floor:
  Var(x_i) ≈ Var(counts_i)/dt^2 (+ Var(bg_counts_i)/dt^2 if provided)
  with Var(counts_i) ≈ expected total counts ≈ observed counts_i (standard plug-in)

Monte Carlo noise band (recommended):
  Null = constant SOURCE rate + your time-varying background model.
  i.e. lambda_i = (mean_source_rate * dt) + bg_counts_i
  Draw counts_i ~ Poisson(lambda_i), compute SF on net rate, build quantiles.

MVT:
  default: smallest τ where SF(τ) > MC upper band (if MC enabled)
  fallback: smallest τ where SF(τ) > analytic floor + nsigma*SE (heuristic)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class SFResult:
    tau: np.ndarray
    sf: np.ndarray
    sf_noise: np.ndarray
    sf_mc_lo: Optional[np.ndarray]
    sf_mc_hi: Optional[np.ndarray]
    mvt: Optional[float]
    details: Dict[str, object]


def calc_structure_function(
    counts: np.ndarray,
    bg_counts: np.ndarray,
    dt: float,
    *,
    max_lag: Optional[float] = None,
    lags: Optional[np.ndarray] = None,
    # Optional background *uncertainty* (variance) in counts/bin from your fit:
    bg_var_counts: Optional[np.ndarray] = None,
    # Monte Carlo options:
    mc_trials: int = 300,
    mc_quantiles: Tuple[float, float] = (0.05, 0.95),
    random_seed: Optional[int] = 0,
    # MVT selection:
    mvt_rule: str = "mc",  # "mc" or "analytic"
    analytic_nsigma: float = 3.0,
    min_pairs_per_lag: int = 20,
    # If your net can be negative, keep it; if you prefer to clip:
    clip_mean_source_rate_at_zero: bool = True,
) -> SFResult:
    """
    Parameters
    ----------
    counts : array
        Observed total counts per bin (source + background).
    bg_counts : array
        Estimated background counts per bin (from prior fit), can vary with time.
    dt : float
        Uniform bin width in seconds.
    max_lag : float, optional
        Maximum lag in seconds. Default: ~1/4 total span.
    lags : array, optional
        Explicit lag values (seconds). Snapped to integer multiples of dt.
    bg_var_counts : array, optional
        Per-bin variance of bg_counts (counts^2). If provided, it is added to Var(x_i).
        If not provided, background is treated as known/deterministic for variance purposes.
    mc_trials : int
        Monte Carlo trials for noise-only SF band. Set 0 to skip.
    mc_quantiles : (lo, hi)
        Quantiles for MC bands, e.g. (0.05, 0.95).        
    mvt_rule : str
        "mc" or "analytic".
    analytic_nsigma : float
        Sigma factor for analytic heuristic threshold.
    min_pairs_per_lag : int
        Require at least this many pairs at a given lag before considering it for MVT.
    clip_mean_source_rate_at_zero : bool
        When constructing the MC null, mean source rate is estimated from (counts-bg)/dt.
        If True, negative mean is clipped to 0.

    Returns
    -------
    SFResult
    """
    C = np.asarray(counts, dtype=float)
    B = np.asarray(bg_counts, dtype=float)

    if C.ndim != 1 or B.ndim != 1 or C.size != B.size:
        raise ValueError("counts and bg_counts must be 1D arrays of equal length.")
    if np.any(C < 0) or np.any(B < 0):
        raise ValueError("counts and bg_counts must be non-negative.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n = C.size
    if n < 5:
        raise ValueError("Need at least ~5 bins.")

    # Uniform time grid implied
    t = np.arange(n, dtype=float) * dt
    total_span = (t[-1] - t[0]) + dt

    # Define lags
    if max_lag is None and lags is None:
        max_lag = 0.25 * total_span

    if lags is None:
        max_lag = float(max_lag)
        max_k = max(1, min(n - 2, int(np.floor(max_lag / dt))))
        ks = np.arange(1, max_k + 1, dtype=int)
        tau = ks * dt
    else:
        tau_in = np.asarray(lags, dtype=float)
        if np.any(tau_in <= 0):
            raise ValueError("All lags must be > 0.")
        ks = np.unique(np.clip(np.rint(tau_in / dt).astype(int), 1, n - 2))
        tau = ks * dt

    # Net rate series for SF
    x = (C - B) / dt  # counts/s

    # -------- Analytic Poisson floor --------
    # Plug-in variance for total counts: Var(C_i) ~ E[C_i] ~ C_i (Poisson)
    var_counts = C.copy()

    # Optional background-fit variance term (counts^2)
    if bg_var_counts is not None:
        bg_var_counts = np.asarray(bg_var_counts, dtype=float)
        if bg_var_counts.shape != C.shape:
            raise ValueError("bg_var_counts must have same shape as counts.")
        if np.any(bg_var_counts < 0):
            raise ValueError("bg_var_counts must be >= 0.")
        var_counts = var_counts + bg_var_counts

    # Var(net rate) in (counts/s)^2
    var_x = var_counts / (dt ** 2)

    sf = np.full_like(tau, np.nan, dtype=float)
    sf_noise = np.full_like(tau, np.nan, dtype=float)
    n_pairs = np.zeros_like(ks, dtype=int)

    for j, k in enumerate(ks):
        dif = x[k:] - x[:-k]
        sf[j] = np.mean(dif * dif)
        n_pairs[j] = dif.size

        # Expected noise contribution for differences: Var(x_{i+k}) + Var(x_i)
        vn = var_x[k:] + var_x[:-k]
        sf_noise[j] = np.mean(vn)

    # -------- Monte Carlo noise-only band (recommended) --------
    sf_mc_lo = None
    sf_mc_hi = None
    if mc_trials and mc_trials > 0:
        rng = np.random.default_rng(random_seed)

        # Estimate mean SOURCE rate from net series (could be negative if bg over-subtracted)
        mean_source_rate = float(np.mean((C - B) / dt))
        if clip_mean_source_rate_at_zero:
            mean_source_rate = max(0.0, mean_source_rate)

        # Null: constant source + your varying background model
        # lambda_i in counts/bin:
        lam = (mean_source_rate * dt) + B
        lam = np.clip(lam, 0.0, None)

        mc_sfs = np.empty((mc_trials, tau.size), dtype=float)

        for m in range(mc_trials):
            C_sim = rng.poisson(lam=lam, size=n).astype(float)
            x_sim = (C_sim - B) / dt

            for j, k in enumerate(ks):
                dif = x_sim[k:] - x_sim[:-k]
                mc_sfs[m, j] = np.mean(dif * dif)

        lo, hi = mc_quantiles
        sf_mc_lo = np.quantile(mc_sfs, lo, axis=0)
        sf_mc_hi = np.quantile(mc_sfs, hi, axis=0)

    # -------- MVT selection --------
    mvt = None
    eligible = n_pairs >= int(min_pairs_per_lag)

    if mvt_rule.lower() == "mc":
        if sf_mc_hi is None:
            raise ValueError("mvt_rule='mc' requires mc_trials > 0.")
        idx = np.where(eligible & (sf > sf_mc_hi))[0]
        if idx.size:
            mvt = float(tau[idx[0]])

    elif mvt_rule.lower() == "analytic":
        # Heuristic: SF > sf_noise + nsigma * SE(mean(vn)).
        # (MC is generally safer; this is a convenience fallback.)
        thresh = np.full_like(sf_noise, np.nan, dtype=float)
        for j, k in enumerate(ks):
            vn = var_x[k:] + var_x[:-k]
            # SE of the mean noise term (not the SF distribution!)
            se = np.std(vn, ddof=1) / np.sqrt(max(1, vn.size))
            thresh[j] = sf_noise[j] + float(analytic_nsigma) * se

        idx = np.where(eligible & np.isfinite(thresh) & (sf > thresh))[0]
        if idx.size:
            mvt = float(tau[idx[0]])
    else:
        raise ValueError("mvt_rule must be 'mc' or 'analytic'.")

    details = dict(
        dt=dt,
        total_span=total_span,
        n_bins=n,
        n_pairs=n_pairs,
        max_lag=float(tau[-1]) if tau.size else None,
        bg_var_used=(bg_var_counts is not None),
        mc_trials=int(mc_trials),
        mc_quantiles=mc_quantiles,
        mvt_rule=mvt_rule,
        analytic_nsigma=float(analytic_nsigma),
        min_pairs_per_lag=int(min_pairs_per_lag),
        clip_mean_source_rate_at_zero=clip_mean_source_rate_at_zero,
    )

    return SFResult(
        tau=tau,
        sf=sf,
        sf_noise=sf_noise,
        sf_mc_lo=sf_mc_lo,
        sf_mc_hi=sf_mc_hi,
        mvt=mvt,
        details=details,
    )


# ---------------- Example usage ----------------
def usage_example(add_flare1=True, add_flare2=False, sigma=0.15, amplitude=250, mc_quantiles=(0.05, 0.999)):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    dt = 0.1
    t = np.arange(0, 100, dt)

    # Varying background (counts/s), plus a flaring source
    bg_rate = 40.0 + 10.0 * np.sin(2 * np.pi * t / 60.0)
    flare1 = amplitude * np.exp(-0.5 * ((t - 50.0) / sigma) ** 2) if add_flare1 else 0.0  
    flare2 = amplitude * np.exp(-0.5 * ((t - 50.0/2) / (sigma/2)) ** 2) if add_flare2 else 0.0  

    src_rate = 5.0 + flare1 + flare2  # counts/s

    true_total_rate = np.clip(bg_rate + src_rate, 0, None)

    # Observed total counts/bin
    counts = rng.poisson(true_total_rate * dt).astype(float)

    # Background model in counts/bin (pretend you fit this)
    bg_counts = (bg_rate * dt).astype(float)

    res = calc_structure_function(
        counts=counts,
        bg_counts=bg_counts,
        dt=dt,
        max_lag=25.0,
        mc_trials=400,
        mc_quantiles=(0.05, 0.995),
        mvt_rule="mc",
        # mvt_rule="analytic",
        # analytic_nsigma=10.0,
        min_pairs_per_lag=50,
        # bg_var_counts=...  # optional if you have per-bin bg uncertainty
        # bg_var_counts=bg_counts**0.5
    )

    print("Estimated MVT (s):", res.mvt)

    # Create two independent subplots in the same figure (no shared axes)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=False, figsize=(8, 6), gridspec_kw={"height_ratios": [1, 2]}
    )

    # Top: counts vs t
    ax1.step(t, counts, where="mid", color="tab:blue", lw=1, label="counts")
    ax1.plot(t, bg_counts, color="tab:orange", lw=1, label="background (counts/bin)", linestyle="--")
    # ax1.axhline(bg_counts, color="tab:gray", linestyle="--", label="background (counts/bin)")
    ax1.set_ylabel("Counts per bin")
    ax1.legend(loc="upper right")
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_xscale("linear")

    # plt.figure()
    ax2.loglog(res.tau, res.sf, marker="o", linestyle="-", lw=1, label="SF (net rate)")
    ax2.loglog(res.tau, res.sf_noise, linestyle="--", lw=1, label="Analytic Poisson floor")
    if res.sf_mc_lo is not None:
        ax2.fill_between(res.tau, res.sf_mc_lo, res.sf_mc_hi, alpha=0.3, label="MC noise band")
    if res.mvt is not None:
        ax2.axvline(res.mvt, linestyle=":", color="k", label=f"MVT ≈ {res.mvt:.3g} s")
    ax2.set_xlabel("Lag τ (s)")
    ax2.set_ylabel("SF(τ) = <(x(t+τ)-x(t))^2>   [x in counts/s]")
    ax2.legend()
    plt.tight_layout()

    slope = np.diff(np.log10(res.sf)) / np.diff(np.log10(res.tau))
    plt.figure(figsize=(8, 4))
    plt.plot(res.tau[1:], slope, marker="o", lw=1)
    plt.xscale("log")
    plt.xlabel("Lag τ (s)")
    plt.ylabel("d log10(SF) / d log10(τ)")
    plt.axhline(0, color="k", linestyle="--", lw=1, alpha=0.5)
    plt.axhline(1, color="k", linestyle="--", lw=1, alpha=0.5)
    plt.axhline(2, color="k", linestyle="--", lw=1, alpha=0.5)
    plt.title("Structure Function Slope vs Lag")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()

    plt.show()
