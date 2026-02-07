import numpy as np
from scipy.signal import get_window
from scipy.stats import chi2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Sequence, Dict
from scipy.signal import correlate, correlation_lags, detrend
from scipy.stats import norm

##### Frequency-domain search (FFT/periodogram) for binned data #####

def leahy_periodogram(x, dt):
    # x: counts per bin (mean >~ few), dt: bin size
    x = np.asarray(x)
    N = len(x)
    mean = np.mean(x)
    x_demean = x - mean
    # FFT (one-sided)
    fft = np.fft.rfft(x_demean)
    # Leahy norm: 2/μ * |FFT|^2 / N  (μ = mean counts per bin)
    P = (2.0/mean) * (np.abs(fft)**2) / N
    freqs = np.fft.rfftfreq(N, dt)
    return freqs, P

def averaged_leahy(x, dt, seglen, window='hann'):
    N = len(x)
    nseg = N // seglen
    P_stack = []
    win = get_window(window, seglen, fftbins=True)
    wnorm = (np.sum(win**2)/seglen)
    for i in range(nseg):
        seg = x[i*seglen:(i+1)*seglen]
        # window, renormalize to preserve Leahy expectation
        segw = seg * win
        f, P = leahy_periodogram(segw, dt)
        # undo window power loss to keep white noise ~2
        P /= wnorm
        P_stack.append(P)
    P_stack = np.array(P_stack)
    return f, P_stack.mean(axis=0), nseg  # nseg = M

# Example significance test at a frequency index k:
def global_pvalue(Pk, cont_k, M, N_trials):
    # Pk: observed power; cont_k: model continuum level at freq k (Leahy units)
    # Under H0, Pk/cont_k ~ (1/(2M)) * χ²_{2M}
    x = (2*M) * (Pk/cont_k)
    p_single = 1.0 - chi2.cdf(x, df=2*M)
    p_global = 1.0 - (1.0 - p_single)**N_trials
    return p_single, p_global

@dataclass
class Peak:
    """Container for a candidate peak in the periodogram.

    Attributes
    ----------
    nu0 : float
        Centroid frequency (Hz).
    width : float
        Full width at half max (Hz). Use for Q = nu0/width.
    power : float
        Observed periodogram power at/near nu0 (Leahy units if using Leahy).
    frac_rms : Optional[float]
        Fractional rms attributed to this peak (if known).
    p_single : Optional[float]
        Single-trial p-value for the peak.
    p_global : Optional[float]
        Trials-corrected (global) p-value for the search band.
    meta : dict
        Any extra fit results (e.g., amplitude, dof, errors).
    """
    nu0: float
    width: float
    power: float
    frac_rms: Optional[float] = None
    p_single: Optional[float] = None
    p_global: Optional[float] = None
    meta: Dict = None

def _ensure_scipy(name: str):
    if name == 'chi2' and chi2 is None:
        raise ImportError("scipy is required for significance/FAP calculations (scipy.stats.chi2)")
    if name == 'correlate' and (correlate is None or correlation_lags is None):
        raise ImportError("scipy is required for ACF (scipy.signal.correlate)")
 
def plot_periodogram(freqs: np.ndarray,
                     power: np.ndarray,
                     ax: Optional[plt.Axes] = None,
                     model_continuum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                     peaks: Optional[Sequence[Peak]] = None,
                     M: Optional[int] = None,
                     N_trials: Optional[int] = None,
                     fmin: Optional[float] = None,
                     fmax: Optional[float] = None,
                     loglog: bool = True,
                     title: Optional[str] = None) -> plt.Axes:
    """Plot an (averaged) Leahy periodogram with optional continuum & FAP bands.

    Parameters
    ----------
    freqs, power : arrays
        One-sided Fourier frequencies (Hz) and corresponding powers (Leahy norm).
    model_continuum : (f_model, P_model), optional
        Continuum model values to overlay (in Leahy units). If provided along
        with M and N_trials, dashed horizontal FAP=1%, 0.1% lines are drawn at
        the *median* continuum level in the displayed band.
    peaks : list[Peak], optional
        Candidate peaks to annotate with ν0 and global p-values.
    M : int, optional
        Number of averaged segments used to form the periodogram (for FAP).
    N_trials : int, optional
        Effective number of independent frequencies scanned (for FAP).
    fmin, fmax : float, optional
        Restrict the displayed frequency range.
    loglog : bool
        Use log–log axes (recommended for red noise). If False, use semilogx.
    title : str, optional
        Figure title.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    # Mask frequency range
    mask = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    f = freqs[mask]
    P = power[mask]

    if loglog:
        ax.loglog(f, P, drawstyle='steps-mid', lw=1.0, label='Periodogram')
    else:
        ax.semilogx(f, P, drawstyle='steps-mid', lw=1.0, label='Periodogram')

    # Overlay continuum model if given
    cont_level_for_fap = None
    if model_continuum is not None:
        fm, Pm = model_continuum
        mmask = np.ones_like(fm, dtype=bool)
        if fmin is not None:
            mmask &= fm >= fmin
        if fmax is not None:
            mmask &= fm <= fmax
        if loglog:
            ax.loglog(fm[mmask], Pm[mmask], color='C1', lw=1.0, label='Continuum')
        else:
            ax.semilogx(fm[mmask], Pm[mmask], color='C1', lw=1.0, label='Continuum')
        cont_level_for_fap = np.median(Pm[mmask]) if np.any(mmask) else np.median(Pm)

    # FAP thresholds
    if (M is not None) and (N_trials is not None) and (cont_level_for_fap is not None):
        for a, ls, lab in [(1e-2, ':', '1% global FAP'), (1e-3, '--', '0.1% global FAP')]:
            thr = fap_threshold_leahy(cont_level_for_fap, M, N_trials, alpha_global=a)
            ax.axhline(thr, linestyle=ls, alpha=0.6, label=f"{lab}")

    # Annotate peaks
    if peaks:
        for pk in peaks:
            ax.axvline(pk.nu0, color='C3', alpha=0.6, lw=1.0)
            txt = f"ν₀={pk.nu0:.3g} Hz\nQ={pk.nu0/max(pk.width, 1e-12):.2g}"
            if pk.p_global is not None:
                txt += f"\np={pk.p_global:.2g}"
            ax.annotate(txt, xy=(pk.nu0, pk.power), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.9))

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Leahy power')
    if title:
        # ax.set_title(title)
        ax.text(0.0175, 0.95, title, transform=ax.transAxes,
                        ha='left', va='top', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='best', framealpha=0.85)
    ax.set_xscale('linear')

    return ax

def ar2_qpo(N, dt, f0, Q=20.0, rng=None):
    """
    Narrow-band stochastic oscillator (AR(2)) centered at f0 (Hz) with quality Q.

    x_t = 2 r cos(2π f0 dt) x_{t-1} - r^2 x_{t-2} + σ ε_t

    r ~ exp(-π f0 dt / Q) controls the linewidth (~ f0 / (2Q)).
    Output has unit variance by construction (we normalize at the end).
    """
    if rng is None:
        rng = np.random.default_rng()
    # pole radius sets coherence (Q)
    r = np.exp(-np.pi * f0 * dt / max(Q, 1e-6))
    theta = 2.0 * np.pi * f0 * dt
    a1 = 2.0 * r * np.cos(theta)
    a2 = -r**2

    # Drive with white noise and normalize to unit variance empirically
    x = np.zeros(N, dtype=np.float64)
    eps = rng.standard_normal(N)
    for t in range(2, N):
        x[t] = a1 * x[t-1] + a2 * x[t-2] + eps[t]
    # remove transient
    x = x[int(5.0 / dt):] if N > int(5.0 / dt) else x
    if x.size < N:
        # pad to length N
        x = np.pad(x, (0, N - x.size))
    # normalize to zero mean, unit std
    x -= np.mean(x)
    std = np.std(x)
    if std > 0:
        x /= std
    return x

def ou_red_noise(N, dt, tau=2.0, rng=None):
    """
    Ornstein-Uhlenbeck (OU) red-noise process with correlation time tau (s).
    Returns zero-mean, unit-variance series.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = np.exp(-dt / max(tau, 1e-9))
    sigma = np.sqrt(1 - alpha**2)  # ensures stationary unit variance with unit-variance driving noise
    x = np.zeros(N, dtype=np.float64)
    eps = rng.standard_normal(N)
    for t in range(1, N):
        x[t] = alpha * x[t-1] + sigma * eps[t]
    x -= np.mean(x)
    std = np.std(x)
    if std > 0:
        x /= std
    return x

def leahy_p_single(Pk: float, continuum: float, M: int) -> float:
    """
    Single-trial tail probability for an M-averaged Leahy power.
    Under H0: (Pk / continuum) ~ (1/(2M)) * chi2_{2M}.
    """
    if M < 1:
        raise ValueError("M must be >= 1")
    x = (2.0 * M) * (Pk / max(continuum, 1e-300))
    return 1.0 - chi2.cdf(x, df=2*M)

def leahy_p_global(Pk: float, continuum: float, M: int, N_trials: int) -> float:
    """
    Familywise/global p-value over N_trials independent frequencies (Bonferroni-like).
    """
    p1 = leahy_p_single(Pk, continuum, M)
    N = max(int(N_trials), 1)
    return 1.0 - (1.0 - p1)**N

def estimate_trials(freqs: np.ndarray,
                    fmin: float | None = None,
                    fmax: float | None = None) -> int:
    """
    Effective number of independent frequencies in [fmin, fmax].
    For standard FFT sampling, the Fourier bins are already independent,
    so we approximate by the number of bins in the masked band.
    """
    mask = np.ones_like(freqs, dtype=bool)
    if fmin is not None: mask &= freqs >= fmin
    if fmax is not None: mask &= freqs <= fmax
    # Exclude DC if present
    if freqs[0] == 0.0:
        mask[0] = False
    return int(np.count_nonzero(mask))

def robust_continuum(power: np.ndarray,
                     freqs: np.ndarray,
                     fmin: float | None = None,
                     fmax: float | None = None,
                     exclude_k: int | None = None,
                     width: int = 3) -> float:
    """
    Robust continuum estimate (Leahy units) as a median in the selected band,
    optionally excluding a small neighborhood around a candidate peak.
    """
    mask = np.ones_like(power, dtype=bool)
    if fmin is not None: mask &= freqs >= fmin
    if fmax is not None: mask &= freqs <= fmax
    if freqs[0] == 0.0:
        mask[0] = False  # drop DC
    if exclude_k is not None:
        i0 = max(exclude_k - width, 0); i1 = min(exclude_k + width + 1, power.size)
        emask = np.ones_like(mask); emask[i0:i1] = False
        mask &= emask
    vals = power[mask]
    if vals.size == 0:
        raise ValueError("No points available to estimate continuum.")
    return float(np.median(vals))

def peak_significance(freqs: np.ndarray,
                      power: np.ndarray,
                      M: int,
                      fmin: float | None = None,
                      fmax: float | None = None,
                      exclude_neighborhood: int = 3) -> dict:
    """
    Compute single-trial and global p-values for the loudest peak in a band.
    Returns a dict with all relevant fields.
    """
    k, f0, Pk, mask = find_peak(freqs, power, fmin=fmin, fmax=fmax)
    cont = robust_continuum(power, freqs, fmin=fmin, fmax=fmax,
                            exclude_k=k, width=exclude_neighborhood)
    Ntr = estimate_trials(freqs, fmin=fmin, fmax=fmax)

    p_single = leahy_p_single(Pk, cont, M)
    p_global = leahy_p_global(Pk, cont, M, Ntr)

    return {
        "k": k, "f0": f0, "Pk": Pk,
        "continuum": cont,
        "M": int(M),
        "N_trials": int(Ntr),
        "p_single": float(p_single),
        "p_global": float(p_global),
        "fap_1pct": float(fap_threshold_leahy(cont, M, Ntr, 1e-2)),
        "fap_0p1pct": float(fap_threshold_leahy(cont, M, Ntr, 1e-3)),
    }

def p_to_sigma(p, one_sided=True):
    """
    Convert a tail probability p into Gaussian sigma significance.
    
    Parameters
    ----------
    p : float or array-like
        Tail probability (p-value). Can be scalar or numpy array.
    one_sided : bool
        If True, treat p as a one-sided tail (typical for detections).
        If False, interpret p as a two-sided probability.

    Returns
    -------
    sigma : float or np.ndarray
        Equivalent Gaussian sigma (number of standard deviations).
    """
    p = np.asarray(p, dtype=float)
    if not one_sided:
        p = p / 2.0  # convert to one-sided before inverting
    # Use the inverse survival function of the standard normal
    sigma = norm.isf(p)
    # Handle underflowed zeros gracefully
    sigma = norm.isf(p)
    # Replace infinities with NaN while preserving scalar vs array return type
    inf_mask = np.isinf(sigma)
    if np.ndim(sigma) == 0:
        # sigma is a scalar-like (0-d array or Python scalar)
        sigma = float(np.nan) if bool(inf_mask) else float(sigma)
    else:
        # sigma is an array
        sigma = np.where(inf_mask, np.nan, sigma)
    return sigma

def fap_threshold_leahy(continuum: float,
                        M: int,
                        N_trials: int,
                        alpha_global: float = 0.01) -> float:
    """Global FAP threshold for Leahy-normalized, M-averaged powers.

    Under H0, P/continuum ~ (1/(2M)) * chi2_{2M}. We want P such that the
    global false alarm (familywise) is alpha_global across N_trials independent
    frequencies. Equivalent single-trial alpha is a = 1 - (1 - alpha_global)^(1/N).

    Parameters
    ----------
    continuum : float
        Continuum level at the frequency of interest (Leahy units).
    M : int
        Number of averaged segments.
    N_trials : int
        Effective number of independent frequencies tested.
    alpha_global : float
        Desired global false-alarm probability.

    Returns
    -------
    float
        Power threshold in Leahy units at which a detection reaches alpha_global.
    """
    _ensure_scipy('chi2')
    if M < 1:
        raise ValueError("M must be >= 1")
    # Convert global alpha to single-trial alpha conservatively (Bonferroni-like)
    alpha_single = 1.0 - (1.0 - alpha_global)**(1.0 / max(N_trials, 1))
    # Tail quantile of chi2_{2M}
    x = chi2.isf(alpha_single, df=2 * M)
    # Scale back to Leahy units
    return continuum * x / (2.0 * M)

def find_peak(freqs: np.ndarray, power: np.ndarray, fmin: float | None = None, fmax: float | None = None):
    mask = np.ones_like(power, dtype=bool)
    if fmin is not None: mask &= freqs >= fmin
    if fmax is not None: mask &= freqs <= fmax
    if freqs[0] == 0.0: mask[0] = False
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("Empty frequency mask.")
    krel = np.argmax(power[idx])
    k = idx[krel]
    return k, float(freqs[k]), float(power[k]), mask

# ---------- Simulated data sets ----------

def simulate_sinusoidal_counts(duration=64.0, dt=0.01,
                                mean_rate=2000.0, mod_freq=5.0, frac_amp=0.1, rng=None):
    if rng is None: rng = np.random.default_rng(123)
    t = np.arange(0, duration, dt)
    lam = mean_rate * (1.0 + frac_amp * np.sin(2*np.pi*mod_freq*t))  # cps
    counts = rng.poisson(lam * dt)
    return t, counts

def simulate_poisson_counts(
    duration=64.0,
    dt=0.01,
    mean_rate=2000.0,         # counts/s  (=> mean per bin = mean_rate*dt)
    qpo_f0=5.0,               # Hz
    qpo_Q=20.0,
    qpo_frac_rms=0.05,        # fractional rms of QPO contribution
    red_tau=None,             # seconds; if None, no red noise
    red_frac_rms=0.02,        # fractional rms of red noise
    rng=None
):
    """
    Build a time series of *Poisson counts per bin* with QPO (+ optional red noise).

    Intensity model:
        lambda(t) = mu * [1 + a_qpo * qpo(t) + a_red * red(t)]
    where qpo(t), red(t) are zero-mean, unit-variance processes scaled to desired
    fractional rms (a_*). We clip to keep intensity positive.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    N = int(np.round(duration / dt))
    t = np.arange(N) * dt
    mu = float(mean_rate)     # counts per second
    mu_bin = mu * dt          # mean counts per bin

    # stochastic QPO
    qpo = ar2_qpo(N, dt, qpo_f0, qpo_Q, rng=rng) if qpo_frac_rms and qpo_frac_rms > 0 else 0.0
    # optional red noise
    red = ou_red_noise(N, dt, tau=red_tau, rng=rng) if red_tau and red_frac_rms > 0 else 0.0

    # combine fractional variations (both zero-mean, unit variance)
    signal = 0.0
    if isinstance(qpo, np.ndarray):
        signal = signal + qpo_frac_rms * qpo
    if isinstance(red, np.ndarray):
        signal = signal + red_frac_rms * red

    # Ensure intensity is positive; small floor keeps Poisson well-defined
    intensity = mu * (1.0 + signal)
    intensity = np.clip(intensity, 1e-6, None)

    # Poisson sample counts per bin:
    counts = rng.poisson(intensity * dt).astype(np.int32)

    return t, counts, {
        "N": N,
        "dt": dt,
        "mean_rate": mean_rate,
        "mean_counts_per_bin": mu_bin,
        "qpo_f0": qpo_f0,
        "qpo_Q": qpo_Q,
        "qpo_frac_rms": qpo_frac_rms,
        "red_tau": red_tau,
        "red_frac_rms": red_frac_rms
    }
    
# ---------- Minimal demos ----------

def test_sinusoidal(duration=64.0, dt=0.064, mean_rate=2000.0, mod_freq=5.0, frac_amp=0.1,
                    rng=None, seg_seconds=4.0, show_average=True, fmin=1.0, fmax=20.0):
    """
    Combined test that simulates a sinusoidal-modulated Poisson light curve,
    computes the single (M=1) Leahy periodogram and an optionally-averaged
    Leahy periodogram, prints significance summaries for both, and plots the
    light curve with both spectra overlaid.

    Parameters
    ----------
    seg_seconds : float
        Segment length (s) used for averaged Leahy; seglen = int(seg_seconds / dt).
    show_average : bool
        If True and averaging is possible, overlay the averaged spectrum on top
        of the single-periodogram.
    fmin, fmax : float
        Search band for significance estimates.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # --- simulate counts ---
    t, counts = simulate_sinusoidal_counts(duration=duration, dt=dt,
                                           mean_rate=mean_rate, mod_freq=mod_freq,
                                           frac_amp=frac_amp, rng=rng)

    # --- single-periodogram ---
    freqs, P_single = leahy_periodogram(counts, dt)

    # --- averaged periodogram (if requested & feasible) ---
    seglen = max(1, int(round(seg_seconds / dt)))
    try:
        f_avg, P_avg, M = averaged_leahy(counts, dt, seglen=seglen, window='hann')
    except Exception:
        # fallback: no averaging possible
        f_avg, P_avg, M = None, None, 0

    if show_average and (M < 1 or P_avg is None):
        # disable averaging if not available
        show_average = False

    # --- significance: averaged (if available) ---
    if show_average:
        res_avg = peak_significance(f_avg, P_avg, M, fmin=fmin, fmax=fmax)
        sig_single_avg = p_to_sigma(res_avg['p_single'], one_sided=True)
        sig_global_avg = p_to_sigma(res_avg['p_global'], one_sided=True)
        print("=== Averaged Leahy (M=%d) peak significance ===" % M)
        print(f"k={res_avg['k']}  f0={res_avg['f0']:.5g} Hz  Pk={res_avg['Pk']:.4g}")
        print(f"continuum~{res_avg['continuum']:.3f}  M={res_avg['M']}  N_trials≈{res_avg['N_trials']}")
        print(f"p_single = {res_avg['p_single']:.3e}")
        print(f"p_global = {res_avg['p_global']:.3e}")
        print(f"Single-trial significance: {sig_single_avg:.2f} σ")
        print(f"Global significance:       {sig_global_avg:.2f} σ")
        print(f"FAP 1% threshold  ≈ {res_avg['fap_1pct']:.3f}")
        print(f"FAP 0.1% threshold ≈ {res_avg['fap_0p1pct']:.3f}")
        print("")

    # --- significance: single periodogram (M=1) ---
    k, f0, Pk, mask = find_peak(freqs, P_single, fmin=fmin, fmax=fmax)
    cont = robust_continuum(P_single, freqs, fmin=fmin, fmax=fmax, exclude_k=k, width=3)
    N_trials = int(np.count_nonzero(mask))
    M1 = 1
    p1 = leahy_p_single(Pk, cont, M=M1)
    pglob = leahy_p_global(Pk, cont, M=M1, N_trials=N_trials)
    thr_1pct = fap_threshold_leahy(cont, M1, N_trials, 1e-2)
    thr_0p1pct = fap_threshold_leahy(cont, M1, N_trials, 1e-3)
    sig_single = p_to_sigma(p1, one_sided=True)
    sig_global = p_to_sigma(pglob, one_sided=True)

    print("=== Single Leahy (M=1) peak significance ===")
    print(f"Peak: f0={f0:.5g} Hz, Pk={Pk:.4g}")
    print(f"Continuum≈{cont:.3f}  (Leahy units, expected ~2 for white noise)")
    print(f"N_trials≈{N_trials},  M={M1}")
    print(f"p_single = {p1:.3e}")
    print(f"p_global = {pglob:.3e}")
    print(f"Single-trial significance: {sig_single:.2f} σ")
    print(f"Global significance:       {sig_global:.2f} σ")
    print(f"FAP 1% threshold  ≈ {thr_1pct:.3f}")
    print(f"FAP 0.1% threshold ≈ {thr_0p1pct:.3f}")

    # --- plotting ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [1, 1.5]})

    # light curve
    ax0.step(t, counts, where='mid', lw=0.8)
    ax0.set_ylabel("Counts / bin")
    ax0.set_xlim(t[0], t[-1])
    ax0.set_title("Simulated light curve (time domain)")
    ax0.grid(True, which='both', alpha=0.25)

    # spectra: plot single always, averaged optionally
    ax1.loglog(freqs[1:], P_single[1:], drawstyle='steps-mid', lw=1.0, label="Single Leahy (M=1)")
    if show_average:
        ax1.loglog(f_avg[1:], P_avg[1:], drawstyle='steps-mid', lw=1.6, color='C1', label=f"Averaged Leahy (M={M})")

    # markers, thresholds
    ax1.axvline(mod_freq, color='C3', alpha=0.6, lw=1.2, label=f"Injected {mod_freq} Hz")
    ax1.axvline(f0, color='C4', lw=1.0, alpha=0.8, label=f"Single peak {f0:.3g} Hz")
    ax1.axhline(thr_1pct, ls=":", alpha=0.7, label="1% global FAP (single)")
    ax1.axhline(thr_0p1pct, ls="--", alpha=0.7, label="0.1% global FAP (single)")
    if show_average:
        # averaged thresholds if available
        ax1.axhline(res_avg["fap_1pct"], ls=":", alpha=0.6, color='C1', label="1% global FAP (avg)")
        ax1.axhline(res_avg["fap_0p1pct"], ls="--", alpha=0.6, color='C1', label="0.1% global FAP (avg)")

    ax1.set_xlim(max(0.1, freqs[1]), max(freqs.max(), f_avg.max() if f_avg is not None else freqs.max()))
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Leahy power")
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_title("Leahy periodogram")
    ax1.set_xscale('linear')

    # plt.tight_layout()
    fig.show()

def test_poisson(show_average=True, f0=5.0):
    """
    Synthetic binned light curve with a QPO for testing Leahy periodograms.

    - Produces Poisson counts per bin with mean >= few (so Leahy is well behaved).
    - QPO is generated via a narrow-band AR(2) stochastic oscillator centered at f0.
    - Optional red noise (OU process) can be mixed in.
    """

    # --- 1) Make synthetic data ---
    duration = 64.0       # s
    dt = 0.064             # s  -> Nyquist = 31.25 Hz
    mean_rate = 2000.0    # counts/s  -> mean per bin ~ 20 counts (healthy for Leahy)
    qpo_f0 = f0          # Hz
    qpo_Q = 25.0
    qpo_frac_rms = 0.10   # 5% fractional rms in intensity
    # red_tau = 2.0         # s (set to None to disable red noise)
    red_tau = None
    red_frac_rms = 0.02

    t, counts, meta = simulate_poisson_counts(duration, dt, mean_rate,
                                      qpo_f0, qpo_Q, qpo_frac_rms,
                                      red_tau, red_frac_rms)

    print("Simulation meta:", meta)
    print(f"Total counts: {counts.sum():,}")

    # --- 2) Compute periodograms using YOUR functions ---
    # Your functions should already be in scope (paste them above this script).
    freqs, P = leahy_periodogram(counts, dt)

    # Averaged periodogram: choose segment length so that f-resolution is adequate
    # seglen in *samples*; e.g., 4 s segments at dt=0.01 -> 400 bins
    seg_seconds = 4.0
    seglen = int(seg_seconds / dt)
    f_avg, P_avg, M = averaged_leahy(counts, dt, seglen, window='hann')

    # --- 3) Plot results ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Light curve
    ax0 = axes[0]
    ax0.step(t, counts, where='mid', lw=0.8)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Counts / bin")
    ax0.set_title("Simulated light curve (Poisson counts)")

    # Periodograms
    ax1 = axes[1]
    # full (single) Leahy periodogram
    ax1.loglog(freqs[1:], P[1:], drawstyle='steps-mid', lw=0.8, label='Leahy (single)')

    if show_average:
        # averaged Leahy (M segments)
        ax1.loglog(f_avg[1:], P_avg[1:], drawstyle='steps-mid', lw=1.5, label=f'Leahy (averaged, M={M})')

    # Quick visual marker at the injected QPO frequency
    ax1.axvline(qpo_f0, color='C3', alpha=0.6, lw=1.2)
    ax1.text(qpo_f0, np.nanmax(P_avg[1:]) * 0.6, f"Injected ~{qpo_f0:g} Hz",
             rotation=90, va='center', ha='left', fontsize=9, color='C3')

    ax1.set_xlim(0.5 * qpo_f0, freqs[1:].max())  # zoom around the QPO region
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Leahy power")
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend()
    ax1.set_title("Leahy periodogram")
    ax1.set_xscale('linear')

    plt.tight_layout()
    plt.show()

def test_gaussian(sigma=0.15, amplitude=250, t1=50, t2=40, t3=60, add_flare1=True, add_flare2=False, 
                  add_flare3=False, show_average=False, vary_bg=False, bg_rate=50, fmin=1.0, fmax=20.0):
    
    rng = np.random.default_rng(1)
    dt = 0.064
    t = np.arange(0, 100, dt)

    # Varying background (counts/s), plus a flaring source
    if vary_bg:
        bg_rate = 40.0 + 10.0 * np.sin(2 * np.pi * t / 60.0)
    else:
        bg_rate = bg_rate * np.ones_like(t)

    flare1 = amplitude * np.exp(-0.5 * ((t - t1) / sigma) ** 2) if add_flare1 else 0.0  
    flare2 = amplitude * np.exp(-0.5 * ((t - t2) / sigma) ** 2) if add_flare2 else 0.0  
    flare3 = amplitude * np.exp(-0.5 * ((t - t3) / sigma) ** 2) if add_flare3 else 0.0  

    src_rate = 5.0 + flare1 + flare2 + flare3  # counts/s
    total_rate = np.clip(bg_rate + src_rate, 0, None)

    # Observed total counts/bin
    counts = rng.poisson(total_rate * dt).astype(float)

    # Background model in counts/bin (pretend you fit this)
    # bg_counts = (bg_rate * dt).astype(float)

    # --- 2) Compute periodograms ---
    freqs, P_single = leahy_periodogram(counts, dt)

    # Averaged periodogram: choose segment length so that f-resolution is adequate
    # seglen in *samples*; e.g., 4 s segments at dt=0.01 -> 400 bins
    seg_seconds = 4.0
    # --- averaged periodogram (if requested & feasible) ---
    seglen = max(1, int(round(seg_seconds / dt)))
    try:
        f_avg, P_avg, M = averaged_leahy(counts, dt, seglen=seglen, window='hann')
    except Exception:
        # fallback: no averaging possible
        f_avg, P_avg, M = None, None, 0

    if show_average and (M < 1 or P_avg is None):
        # disable averaging if not available
        show_average = False

    # --- significance: averaged (if available) ---
    if show_average:
        res_avg = peak_significance(f_avg, P_avg, M, fmin=fmin, fmax=fmax)
        sig_single_avg = p_to_sigma(res_avg['p_single'], one_sided=True)
        sig_global_avg = p_to_sigma(res_avg['p_global'], one_sided=True)
        print("=== Averaged Leahy (M=%d) peak significance ===" % M)
        print(f"k={res_avg['k']}  f0={res_avg['f0']:.5g} Hz  Pk={res_avg['Pk']:.4g}")
        print(f"continuum~{res_avg['continuum']:.3f}  M={res_avg['M']}  N_trials≈{res_avg['N_trials']}")
        print(f"p_single = {res_avg['p_single']:.3e}")
        print(f"p_global = {res_avg['p_global']:.3e}")
        print(f"Single-trial significance: {sig_single_avg:.2f} σ")
        print(f"Global significance:       {sig_global_avg:.2f} σ")
        print(f"FAP 1% threshold  ≈ {res_avg['fap_1pct']:.3f}")
        print(f"FAP 0.1% threshold ≈ {res_avg['fap_0p1pct']:.3f}")
        print("")

    # --- significance: single periodogram (M=1) ---
    k, f0, Pk, mask = find_peak(freqs, P_single, fmin=fmin, fmax=fmax)
    cont = robust_continuum(P_single, freqs, fmin=fmin, fmax=fmax, exclude_k=k, width=3)
    N_trials = int(np.count_nonzero(mask))
    M1 = 1
    p1 = leahy_p_single(Pk, cont, M=M1)
    pglob = leahy_p_global(Pk, cont, M=M1, N_trials=N_trials)
    thr_1pct = fap_threshold_leahy(cont, M1, N_trials, 1e-2)
    thr_0p1pct = fap_threshold_leahy(cont, M1, N_trials, 1e-3)
    sig_single = p_to_sigma(p1, one_sided=True)
    sig_global = p_to_sigma(pglob, one_sided=True)

    print("=== Single Leahy (M=1) peak significance ===")
    print(f"Peak: f0={f0:.5g} Hz, Pk={Pk:.4g}")
    print(f"Continuum≈{cont:.3f}  (Leahy units, expected ~2 for white noise)")
    print(f"N_trials≈{N_trials},  M={M1}")
    print(f"p_single = {p1:.3e}")
    print(f"p_global = {pglob:.3e}")
    print(f"Single-trial significance: {sig_single:.2f} σ")
    print(f"Global significance:       {sig_global:.2f} σ")
    print(f"FAP 1% threshold  ≈ {thr_1pct:.3f}")
    print(f"FAP 0.1% threshold ≈ {thr_0p1pct:.3f}")


    # --- 3) Plot results ---
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [1, 1.5]})
    
    # light curve
    ax0.step(t, counts, where='mid', lw=0.8)
    ax0.set_ylabel("Counts / bin")
    ax0.set_xlim(t[0], t[-1])
    ax0.set_title("Simulated light curve (Poisson background + Gaussian flares)")
    ax0.grid(True, which='both', alpha=0.25)

    # spectra: plot single always, averaged optionally
    ax1.loglog(freqs[1:], P_single[1:], drawstyle='steps-mid', lw=1.0, label="Single Leahy (M=1)")
    if show_average:
        ax1.loglog(f_avg[1:], P_avg[1:], drawstyle='steps-mid', lw=1.6, color='C1', label=f"Averaged Leahy (M={M})")

    # markers, thresholds
    # ax1.axvline(mod_freq, color='C3', alpha=0.6, lw=1.2, label=f"Injected {mod_freq} Hz")
    ax1.axvline(f0, color='C4', lw=1.0, alpha=0.8, label=f"Single peak {f0:.3g} Hz")
    ax1.axhline(thr_1pct, ls=":", alpha=0.7, label="1% global FAP (single)")
    ax1.axhline(thr_0p1pct, ls="--", alpha=0.7, label="0.1% global FAP (single)")
    if show_average:
        # averaged thresholds if available
        ax1.axhline(res_avg["fap_1pct"], ls=":", alpha=0.6, color='C1', label="1% global FAP (avg)")
        ax1.axhline(res_avg["fap_0p1pct"], ls="--", alpha=0.6, color='C1', label="0.1% global FAP (avg)")

    ax1.set_xlim(max(0.1, freqs[1]), max(freqs.max(), f_avg.max() if f_avg is not None else freqs.max()))
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Leahy power")
    ax1.grid(True, which='both', alpha=0.25)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_title("Leahy periodogram")
    ax1.set_xscale('linear')

    plt.tight_layout()
    plt.show()

# ---------- Search routine ----------

def search(time, counts, dt, fmin=1.0, fmax=20.0, title=None, detector_label=None, energy_label=None):

    freqs, P = leahy_periodogram(counts, dt)

    # ---- pick a search band and compute significance with M=1 ----
    k, f0, Pk, mask = find_peak(freqs, P, fmin=fmin, fmax=fmax)
    cont = robust_continuum(P, freqs, fmin=fmin, fmax=fmax, exclude_k=k, width=3)
    N_trials = int(np.count_nonzero(mask))    # approx. independent bins in band
    M = 1                                     # single (non-averaged) spectrum

    p1 = leahy_p_single(Pk, cont, M=M)
    pglob = leahy_p_global(Pk, cont, N_trials=N_trials, M=M)
    thr_1pct = fap_threshold_leahy(cont, M, N_trials, 1e-2)
    thr_0p1pct = fap_threshold_leahy(cont, M, N_trials, 1e-3)

    # Convert the p-values to sigma significances
    sig_single = p_to_sigma(p1, one_sided=True)
    sig_global = p_to_sigma(pglob, one_sided=True)

    print("=== Leahy significance ===")
    print(f"Peak: f0={f0:.5g} Hz, Pk={Pk:.4g}")
    print(f"Continuum≈{cont:.3f}  (Leahy units, expected ~2 for white noise)")
    print(f"N_trials≈{N_trials},  M={M}")
    print(f"p_single = {p1:.3e}")
    print(f"p_global = {pglob:.3e}")
    print(f"Single-trial significance: {sig_single:.2f} σ")
    print(f"Global significance:       {sig_global:.2f} σ")
    print(f"FAP 1% threshold  ≈ {thr_1pct:.3f}")
    print(f"FAP 0.1% threshold ≈ {thr_0p1pct:.3f}")

    # ---- plot: light curve above periodogram ----
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [1, 1.3]})
    
    if title is not None:
        fig.canvas.manager.set_window_title(title)

    # Light curve panel
    ax0.step(time, counts, where='mid', lw=0.8)
    ax0.set_ylabel("Counts / bin")
    ax0.set_title("Input light curve")
    ax0.grid(True, which="both", alpha=0.25)
    ax0.set_xlim(time[0], time[-1])

    # Add a detector and energy labels, if provided
    if detector_label is not None:
        ax0.text(0.0175, 0.95, detector_label, transform=ax0.transAxes,
                ha='left', va='top', fontsize=10)
    if energy_label is not None:
        ax0.text(max(0.01, 1.0 - 0.00925 * len(energy_label)), 0.95, energy_label, 
        transform=ax0.transAxes, ha='left', va='top', fontsize=10)

    # Periodogram panel
    ax1.loglog(freqs[1:], P[1:], drawstyle='steps-mid', lw=1.2, label="Leahy (single)")
    ax1.axvline(f0, color='C3', lw=1.0, alpha=0.8)
    ax1.axhline(thr_1pct, ls=":", alpha=0.8, label="1% global FAP")
    ax1.axhline(thr_0p1pct, ls="--", alpha=0.8, label="0.1% global FAP")
    ax1.set_xlim(fmin, freqs[1:].max())
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Leahy power")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend()
    ax1.set_title(f"Peak at {f0:.3f} Hz; p_global={pglob:.2e}  (single spectrum, M=1)")
    ax1.set_xscale('linear')

    fig.tight_layout()
    fig.show()      