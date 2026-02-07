import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Sequence

from scipy.fft import rfft, rfftfreq


@dataclass
class FFTAnalysisResult:
    freq_hz: np.ndarray
    fft: np.ndarray
    psd_raw: np.ndarray
    psd_rms2_per_hz: np.ndarray
    psd_leahy: np.ndarray
    mean_rate: float
    dt: float
    n: int
    meta: Dict[str, object]


def fft_rate_analysis(
    rate: np.ndarray,
    dt: float,
    *,
    detrend: str = "mean",            # "none" | "mean" | "linear"
    window: str = "hann",             # "none" | "hann" | "hamming" | "blackman"
    normalize: str = "rms2_per_hz",   # "raw" | "rms2_per_hz" | "leahy"
    nfft: Optional[int] = None,
    plot: bool = True,
    title: Optional[str] = None,
    max_freq: Optional[float] = None,
    ax: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> FFTAnalysisResult:
    """
    FFT analysis for evenly spaced count-rate time series using scipy.fft.

    normalize:
      - "raw": |FFT|^2 (units depend on scaling)
      - "rms2_per_hz": one-sided PSD in fractional variance units (rms^2/Hz)
      - "leahy": Leahy-normalized periodogram (Poisson noise ~ 2)
    """
    x_rate = np.asarray(rate, dtype=float)
    if x_rate.ndim != 1:
        raise ValueError("rate must be a 1D array")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite float")

    n0 = x_rate.size
    mean_rate = float(np.mean(x_rate))

    # Build the series to FFT, depending on normalization:
    # - For Leahy, the canonical input is counts/bin.
    # - For RMS, rate is fine (we compute rms^2/Hz and fractional rms^2/Hz).
    x_counts = x_rate * dt
    mean_counts = float(np.mean(x_counts))

    # Detrend helper
    def apply_detrend(y: np.ndarray, y_mean: float) -> np.ndarray:
        if detrend == "none":
            return y.copy()
        if detrend == "mean":
            return y - y_mean
        if detrend == "linear":
            t = np.arange(n0) * dt
            p = np.polyfit(t, y, deg=1)
            trend = np.polyval(p, t)
            return y - trend
        raise ValueError("detrend must be one of: 'none','mean','linear'")

    y_rate = apply_detrend(x_rate, mean_rate)
    y_counts = apply_detrend(x_counts, mean_counts)

    # Window
    if window == "none":
        w = np.ones(n0, dtype=float)
    elif window == "hann":
        w = np.hanning(n0)
    elif window == "hamming":
        w = np.hamming(n0)
    elif window == "blackman":
        w = np.blackman(n0)
    else:
        raise ValueError("window must be one of: 'none','hann','hamming','blackman'")

    # Choose FFT input depending on normalization
    y = y_counts if normalize == "leahy" else y_rate
    yw = y * w

    # NFFT
    if nfft is None:
        n = n0
    else:
        n = int(nfft)
        if n <= 0:
            raise ValueError("nfft must be a positive integer")
        if n > n0:
            yw = np.pad(yw, (0, n - n0), mode="constant", constant_values=0.0)
        elif n < n0:
            yw = yw[:n]

    # FFT (one-sided for real input)
    X = rfft(yw)
    f = rfftfreq(n, dt)

    # Raw power
    Praw = np.abs(X) ** 2

    # Window power correction U = mean(w^2), adjusted if padded/truncated
    if n == n0:
        U = float(np.mean(w ** 2))
    else:
        ww = w
        if n > n0:
            ww = np.pad(ww, (0, n - n0), mode="constant", constant_values=0.0)
        else:
            ww = ww[:n]
        U = float(np.mean(ww ** 2))

    # RMS-style PSD in absolute rms^2/Hz (one-sided, with DC/Nyq halved)
    psd_abs = (2.0 * dt) / (n * U) * Praw
    psd_abs[0] *= 0.5
    if n % 2 == 0 and psd_abs.size > 1:
        psd_abs[-1] *= 0.5

    # Fractional rms^2/Hz (what your original code plotted)
    psd_frac = psd_abs / (mean_rate ** 2) if mean_rate != 0.0 else psd_abs.copy()

    # Leahy normalization (Poisson noise ~ 2), using total counts (un-detrended)
    # IMPORTANT: do NOT apply the one-sided doubling here.
    # We include window correction /U to keep the expected noise level near 2 when windowing.
    Nph = float(np.sum(x_counts))  # total photons in the segment (from original series)
    if Nph > 0:
        P_leahy = (2.0 / (Nph * U)) * Praw
    else:
        P_leahy = np.full_like(Praw, np.nan, dtype=float)

    # Plot
    if plot:
        if ax is None:
            fig = plt.figure(figsize=(10,8))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
        else:
            ax1, ax2 = ax

        tplot = np.arange(n0) * dt
        ax1.plot(tplot, x_rate, lw=1)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rate (counts/s)")
        if title:
            ax1.set_title(title)

        if normalize == "leahy":
            yplot = P_leahy
            ylabel = "Leahy power"
        elif normalize == "rms2_per_hz":
            yplot = psd_frac
            ylabel = "PSD (fractional rms$^2$/Hz)"
        elif normalize == "raw":
            yplot = Praw
            ylabel = r"$|FFT|^2$ (raw)"
        else:
            raise ValueError("normalize must be 'raw', 'rms2_per_hz', or 'leahy'")

        # Skip DC for log scales
        ax2.plot(f[1:], yplot[1:], lw=1)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel(ylabel)
        ax2.set_xscale("linear")
        ax2.set_yscale("log")
        if max_freq is not None:
            ax2.set_xlim(left=f[1], right=max_freq)
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.show()

    return FFTAnalysisResult(
        freq_hz=f,
        fft=X,
        psd_raw=Praw,
        psd_rms2_per_hz=psd_frac,
        psd_leahy=P_leahy,
        mean_rate=mean_rate,
        dt=dt,
        n=n,
        meta={
            "detrend": detrend,
            "window": window,
            "normalize": normalize,
            "U_window_power": U,
            "nfft": nfft,
            "Nph_total_counts": Nph,
        },
    )

def leahy_periodogram(counts: np.ndarray, dt: float):
    """
    Compute Leahy-normalized periodogram for evenly binned counts.

    Leahy normalization: P_k = (2/N_ph) * |FFT_k|^2
      where FFT is of the *counts* time series (not mean-subtracted),
      and N_ph is the total number of counts (sum of counts).
    Noise expectation for pure Poisson is ~2 at all frequencies (excluding DC).

    Returns
    -------
    freqs : ndarray
        Positive Fourier frequencies (Hz), excluding DC.
    power : ndarray
        Leahy power at those frequencies.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1:
        raise ValueError("counts must be a 1D array.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")

    n = counts.size
    n_ph = counts.sum()
    if n_ph <= 0:
        raise ValueError("Total counts must be > 0 for Leahy normalization.")

    # rFFT gives [0..Nyquist]
    fft = np.fft.rfft(counts)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Leahy power including DC; we'll drop DC afterward
    power = (2.0 / n_ph) * (np.abs(fft) ** 2)

    # Exclude DC (freq=0)
    return freqs[1:], power[1:]


def leahy_significance_level(p_false: float, dof: int = 2) -> float:
    """
    Threshold z such that P(Power > z) = p_false for chi-square(dof) noise.

    For standard Leahy (no averaging), dof=2 and z = -2 ln(p_false).
    """
    p_false = float(p_false)
    if not (0 < p_false < 1):
        raise ValueError("p_false must be in (0, 1).")
    if dof == 2:
        return -2.0 * np.log(p_false)

    # Generic chi-square inverse survival function without scipy:
    # If you need dof != 2, install scipy and use scipy.stats.chi2.isf.
    raise NotImplementedError(
        "dof != 2 requires scipy (use scipy.stats.chi2.isf)."
    )


def trials_corrected_p(p_global: float, n_trials: int) -> float:
    """
    Convert a desired global false-alarm probability (over many frequencies)
    into a per-trial p-value using:
        p_global = 1 - (1 - p_single)^n_trials
      => p_single = 1 - (1 - p_global)^(1/n_trials)
    """
    p_global = float(p_global)
    if not (0 < p_global < 1):
        raise ValueError("p_global must be in (0, 1).")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    # numerically stable form:
    return 1.0 - np.exp(np.log1p(-p_global) / n_trials)


def add_leahy_significance_lines(
    ax: plt.Axes,
    n_trials: int,
    *,
    single_trial_alphas=(0.01, 0.001),
    global_alphas=(0.01, 0.001),
    dof: int = 2,
):
    """
    Overlay horizontal significance lines on an existing Leahy power plot.

    Parameters
    ----------
    ax : matplotlib Axes
    n_trials : int
        Number of independent frequencies searched. A common choice for rFFT is
        len(freqs) (i.e., N/2 for even N) after excluding DC.
    single_trial_alphas : iterable
        Per-frequency false-alarm probabilities to draw (no trials correction).
    global_alphas : iterable
        Global false-alarm probabilities to draw (Bonferroni-like exact correction).
    dof : int
        Degrees of freedom for noise powers (2 for standard Leahy).
    """
    # Single-trial thresholds
    for a in single_trial_alphas:
        z = leahy_significance_level(a, dof=dof)
        ax.axhline(z, linestyle="--", linewidth=1.0,
                   label=f"single-trial p={a:g}  (z={z:.2f})")

    # Global thresholds (trial-corrected)
    for a in global_alphas:
        p_single = trials_corrected_p(a, n_trials)
        z = leahy_significance_level(p_single, dof=dof)
        ax.axhline(z, linestyle=":", linewidth=1.2,
                   label=f"global p={a:g} (n={n_trials}, z={z:.2f})")


def plot_leahy_with_significance(
    counts: np.ndarray,
    dt: float,
    *,
    single_trial_alphas=(0.01, 0.001),
    global_alphas=(0.01, 0.001),
):
    freqs, power = leahy_periodogram(counts, dt)
    n_trials = len(freqs)  # independent rFFT bins excluding DC

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(freqs, power, linewidth=1.0)

    # Optional: typical to draw the expected noise level at 2
    ax.axhline(2.0, linewidth=1.0, label="Poisson noise mean (=2)")

    add_leahy_significance_lines(
        ax,
        n_trials=n_trials,
        single_trial_alphas=single_trial_alphas,
        global_alphas=global_alphas,
        dof=2,
    )

    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Leahy Power")
    ax.set_title("Leahy-normalized Periodogram with Significance Levels")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def demo_fft_rate_analysis(seed: int = 7) -> None:
    """
    Demonstration with simulated Poisson counts from a sinusoidally-modulated rate.
    Produces two plots: fractional-rms PSD and Leahy periodogram.
    """
    rng = np.random.default_rng(seed)

    dt = 0.01
    T = 20.0
    n = int(T / dt)
    t = np.arange(n) * dt

    mean_rate = 200.0
    frac_amp = 0.2
    f0 = 3.0

    rate_true = mean_rate * (1.0 + frac_amp * np.sin(2 * np.pi * f0 * t))
    counts = rng.poisson(rate_true * dt)
    rate_obs = counts / dt

    # Raw PSD (normalization depends on scaling)
    fft_rate_analysis(
        rate_obs,
        dt,
        detrend="mean",
        window="hann",
        normalize="raw",
        plot=True,
        title=f"FFT demo (raw): injected {f0} Hz sinusoid",
        max_freq=50.0,
    )

    # Fractional rms^2/Hz PSD
    fft_rate_analysis(
        rate_obs,
        dt,
        detrend="mean",
        window="hann",
        normalize="rms2_per_hz",
        plot=True,
        title=f"FFT demo (fractional rms): injected {f0} Hz sinusoid",
        max_freq=50.0,
    )

    # Leahy periodogram
    fft_rate_analysis(
        rate_obs,
        dt,
        detrend="none",   # common choice for Leahy; DC will be huge but we don't plot it
        window="hann",
        normalize="leahy",
        plot=True,
        title=f"FFT demo (Leahy): injected {f0} Hz sinusoid (Poisson noise ~ 2)",
        max_freq=50.0,
    )


# ---- Example usage ----
def leahy_demo(dt=0.064, tmax=100.0, mean_rate=2000.0, amp=5.0, f0=5.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = 0.064
    t = np.arange(0, tmax, dt)

    # Example: Poisson noise + weak sinusoid
    # mean_rate = 50.0  # counts/s
    # amp = 5.0         # counts/s
    # f0 = 2.0          # Hz
    rate = mean_rate + amp * np.sin(2 * np.pi * f0 * t)

    counts = rng.poisson(rate * dt)

    # import leahy
    # t, counts = leahy.simulate_sinusoidal_counts(duration=64, dt=0.064, mean_rate=2000, mod_freq=5.0,
    #                                              frac_amp=0.1, rng=None)

    fig, ax = plot_leahy_with_significance(
        counts, dt,
        single_trial_alphas=(0.01, 0.001),
        global_alphas=(0.01, 0.001),
    )
    plt.show()

# ---- assumes your fft_rate_analysis() + FFTAnalysisResult exist from earlier ----
# (and that FFTAnalysisResult has: freq_hz, psd_rms2_per_hz, psd_leahy, dt, mean_rate, ...)

# -----------------------------
# Monte Carlo noise-band results
# -----------------------------
@dataclass
class NoiseBandResult:
    freq_hz: np.ndarray                 # rfftfreq grid (includes DC)
    p50: np.ndarray                     # median noise PSD at each f
    p90: np.ndarray
    p95: np.ndarray
    p99: np.ndarray
    n_mc: int
    normalization: str
    meta: Dict[str, object]


def mc_noise_band_from_model(
    *,
    dt: float,
    bg_counts: np.ndarray,              # background model, counts/bin (may vary with time)
    src_rate_model: np.ndarray,         # source model, counts/s (may vary with time)
    n_mc: int = 500,
    normalization: str = "rms2_per_hz", # "rms2_per_hz" | "leahy"
    detrend: str = "mean",
    window: str = "hann",
    nfft: Optional[int] = None,
    percentiles: Sequence[float] = (50, 90, 95, 99),
    seed: Optional[int] = 0,
) -> NoiseBandResult:
    """
    Generate a Monte Carlo noise band for a PSD by simulating Poisson data under a null model.

    Null model:
      counts_i ~ Poisson( lambda_i )
      lambda_i = src_rate_model[i]*dt + bg_counts[i]

    This captures:
      - time-varying background
      - (optional) time-varying source *model* (if you want your null to include it)
      - Poisson statistics
      - your exact FFT/PSD pipeline (windowing, detrending, normalization)

    Notes:
      - If your goal is "noise-only" for MVT, usually set src_rate_model to a CONSTANT
        (e.g., np.full(n, mean_source_rate)) so intrinsic variability is excluded.
      - If you pass a time-varying src_rate_model, the band will include that modeled
        variability (useful for forward-model tests, but not a strict noise-only null).
    """
    bg_counts = np.asarray(bg_counts, dtype=float)
    src_rate_model = np.asarray(src_rate_model, dtype=float)
    if bg_counts.shape != src_rate_model.shape:
        raise ValueError("bg_counts and src_rate_model must have the same shape (n_bins,)")

    n = bg_counts.size
    if n < 8:
        raise ValueError("Need at least ~8 bins for a meaningful PSD")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    lam = src_rate_model * dt + bg_counts
    lam = np.clip(lam, 0.0, None)

    rng = np.random.default_rng(seed)

    # We'll store PSDs for each MC realization on the same frequency grid
    psd_stack = None
    freq_ref = None

    for j in range(n_mc):
        counts_sim = rng.poisson(lam)
        # Convert to rate for fft_rate_analysis input
        rate_sim = counts_sim / dt

        res = fft_rate_analysis(
            rate_sim,
            dt,
            detrend=detrend,
            window=window,
            normalize=normalization,  # "rms2_per_hz" or "leahy"
            nfft=nfft,
            plot=False,
        )

        if normalization == "rms2_per_hz":
            psd_j = res.psd_rms2_per_hz
        elif normalization == "leahy":
            psd_j = res.psd_leahy
        else:
            raise ValueError("normalization must be 'rms2_per_hz' or 'leahy'")

        if psd_stack is None:
            freq_ref = res.freq_hz
            psd_stack = np.empty((n_mc, psd_j.size), dtype=float)

        psd_stack[j, :] = psd_j

    # Compute requested percentile curves frequency-by-frequency
    pct = np.array(percentiles, dtype=float)
    curves = np.percentile(psd_stack, pct, axis=0)

    # Map to named outputs (require the common set)
    def get_curve(p):
        idx = int(np.where(pct == p)[0][0])
        return curves[idx, :]

    out = NoiseBandResult(
        freq_hz=freq_ref,
        p50=get_curve(50) if 50 in pct else np.percentile(psd_stack, 50, axis=0),
        p90=get_curve(90) if 90 in pct else np.percentile(psd_stack, 90, axis=0),
        p95=get_curve(95) if 95 in pct else np.percentile(psd_stack, 95, axis=0),
        p99=get_curve(99) if 99 in pct else np.percentile(psd_stack, 99, axis=0),
        n_mc=n_mc,
        normalization=normalization,
        meta={
            "dt": dt,
            "detrend": detrend,
            "window": window,
            "nfft": nfft,
            "seed": seed,
            "percentiles": list(percentiles),
            "note": "If src_rate_model is time-varying, the band includes that modeled variability.",
        },
    )
    return out


# -----------------------------
# MVT using a noise-band curve
# -----------------------------
@dataclass
class MVTResult:
    mvt: float
    f_max: float
    method: str
    threshold_desc: str
    n_freq_used: int
    meta: Dict[str, object]


def mvt_from_psd_with_noise_band(
    fft_result,
    noise_band: NoiseBandResult,
    *,
    normalization: str = "rms2_per_hz",  # must match noise_band.normalization
    band: str = "p99",                   # "p95" | "p99"
    min_consecutive_bins: int = 1,
    f_min: Optional[float] = None,
    f_max_limit: Optional[float] = None,
    plot: bool = False,
    title: Optional[str] = None,
) -> MVTResult:
    """
    Compute MVT using an MC-derived noise band:
      significant(f) <=> PSD_obs(f) > band_curve(f)
      f_max = highest significant frequency
      MVT = 1 / f_max

    Requires that fft_result and noise_band share the same frequency grid (same dt & nfft).
    """
    if noise_band.normalization != normalization:
        raise ValueError("noise_band.normalization must match normalization")

    freq = fft_result.freq_hz
    if normalization == "rms2_per_hz":
        psd_obs = fft_result.psd_rms2_per_hz
        method = "PSD (fractional RMS)"
    elif normalization == "leahy":
        psd_obs = fft_result.psd_leahy
        method = "Leahy periodogram"
    else:
        raise ValueError("normalization must be 'rms2_per_hz' or 'leahy'")

    if not np.array_equal(freq, noise_band.freq_hz):
        raise ValueError(
            "Frequency grids do not match. Ensure you used the same dt, nfft, and pipeline "
            "for fft_result and noise_band generation."
        )

    band_curve = getattr(noise_band, band)

    # Exclude DC for thresholding & log plotting sanity
    freq2 = freq[1:]
    psd2 = psd_obs[1:]
    band2 = band_curve[1:]

    mask = np.ones_like(freq2, dtype=bool)
    if f_min is not None:
        mask &= freq2 >= f_min
    if f_max_limit is not None:
        mask &= freq2 <= f_max_limit

    freq2 = freq2[mask]
    psd2 = psd2[mask]
    band2 = band2[mask]

    sig = psd2 > band2

    # Optional consecutive-bin requirement
    if min_consecutive_bins > 1:
        k = min_consecutive_bins
        sig_conv = np.convolve(sig.astype(int), np.ones(k, dtype=int), mode="same")
        sig = sig_conv >= k

    if not np.any(sig):
        if plot:
            _plot_psd_vs_band(freq2, psd2, band2, title or f"MVT failed ({method})")
        return MVTResult(
            mvt=np.nan,
            f_max=np.nan,
            method=method,
            threshold_desc=f"obs > {band} noise band",
            n_freq_used=len(freq2),
            meta={"reason": "No bins exceed the noise band", "band": band, "min_consecutive_bins": min_consecutive_bins},
        )

    f_max = float(np.max(freq2[sig]))
    mvt = 1.0 / f_max

    if plot:
        _plot_psd_vs_band(
            freq2, psd2, band2,
            title or f"MVT={mvt:.4g}s (f_max={f_max:.4g} Hz) [{method}]"
        )

    return MVTResult(
        mvt=mvt,
        f_max=f_max,
        method=method,
        threshold_desc=f"obs > {band} noise band",
        n_freq_used=len(freq2),
        meta={
            "band": band,
            "min_consecutive_bins": min_consecutive_bins,
            "f_min": f_min,
            "f_max_limit": f_max_limit,
            "n_mc": noise_band.n_mc,
        },
    )


def _plot_psd_vs_band(freq, psd, band, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freq, psd, lw=1, label="Observed PSD")
    ax.plot(freq, band, lw=1, label="Noise band")
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Example usage (plug-in)
# -----------------------------
def demo_mc_band_and_mvt(seed: int = 7, sigma=0.15, amplitude=250, t1=50, t2=40, t3=60, 
                         add_flare1=True, add_flare2=False, add_flare3=False, show_average=False, 
                         vary_bg=False, bg_rate=1000, fmin=1.0, fmax=20.0):
    """
    Demo:
      - Simulate a sinusoid + background
      - Build MC noise band using a CONSTANT source null (mean rate)
      - Compute MVT from observed PSD crossing the p99 band
    """

    rng = np.random.default_rng(seed)

    dt = 0.01
    T = 20.0
    n = int(T / dt)
    t = np.arange(n) * dt

    # "True" source variability
    mean_src = 2000.0
    frac_amp = 0.2
    f0 = 3.0
    # src_true = mean_src * (1.0 + frac_amp * np.sin(2 * np.pi * f0 * t))
    src_true = mean_src * np.exp(-0.5 * ((t - np.max(t)/2.) / sigma) ** 2) if add_flare1 else 0.0  

    # Modeled background counts/bin (example: slow drift + constant)
    if vary_bg:
        bg_rate = 1000.0 * (1.0 + 0.2 * np.sin(2*np.pi*0.1*t))  # counts/s
    else:
        bg_rate = bg_rate  # constant background rate
    bg_counts = bg_rate * np.ones_like(t) * dt


    # Observed counts and rate
    lam_true = (src_true * dt) + bg_counts
    counts_obs = rng.poisson(lam_true)
    rate_obs = counts_obs / dt

    # Observed PSD
    fft_res = fft_rate_analysis(
        rate_obs,
        dt,
        detrend="mean",
        window="hann",
        normalize="rms2_per_hz",
        plot=True,
        title="Simulated time series"
        # max_freq=20.0,
    )

    # Null source model for MC band: CONSTANT source = mean of your modeled/observed source
    # (This is the usual "noise-only" null for MVT.)
    src_null = np.full(n, np.mean(src_true))

    band = mc_noise_band_from_model(
        dt=dt,
        bg_counts=bg_counts,
        src_rate_model=src_null,
        n_mc=400,
        normalization="rms2_per_hz",
        detrend="mean",
        window="hann",
        nfft=None,
        seed=123,
    )

    mvt = mvt_from_psd_with_noise_band(
        fft_res,
        band,
        normalization="rms2_per_hz",
        band="p99",
        min_consecutive_bins=2,
        f_min=1.0,          # optional: avoid very-low-f leakage region
        f_max_limit=20.0,
        plot=True,
        title="PSD vs MC p99 noise band",
    )

    print("MVT:", mvt.mvt, "s  (f_max:", mvt.f_max, "Hz)")
    # print(mvt)
    return fft_res, band, mvt



if __name__ == "__main__":
    demo_fft_rate_analysis()
