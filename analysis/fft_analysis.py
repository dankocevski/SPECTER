import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from scipy.fft import rfft, rfftfreq


@dataclass
class FFTAnalysisResult:
    freq_hz: np.ndarray
    fft: np.ndarray
    psd_raw: np.ndarray
    psd_rms2_per_hz: np.ndarray
    mean_rate: float
    dt: float
    n: int
    meta: Dict[str, object]


def fft_rate_analysis(
    rate: np.ndarray,
    dt: float,
    *,
    detrend: str = "mean",          # "none" | "mean" | "linear"
    window: str = "hann",           # "none" | "hann" | "hamming" | "blackman"
    normalize: str = "rms2_per_hz", # "raw" | "rms2_per_hz"
    nfft: Optional[int] = None,
    plot: bool = True,
    title: Optional[str] = None,
    max_freq: Optional[float] = None,
    ax: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> FFTAnalysisResult:
    """
    FFT analysis for evenly spaced count-rate time series using scipy.fft.

    Parameters
    ----------
    rate : array
        Count rate time series [counts/s], evenly sampled.
    dt : float
        Sampling interval [s].
    detrend : {"none","mean","linear"}
        Remove constant mean (recommended) or a best-fit line.
    window : {"none","hann","hamming","blackman"}
        Apply a taper to reduce spectral leakage.
    normalize : {"raw","rms2_per_hz"}
        - "raw": |FFT|^2 (units depend on scaling)
        - "rms2_per_hz": one-sided PSD in fractional variance units (rms^2/Hz)
    nfft : int, optional
        Zero-pad/truncate to this length before FFT. Default uses len(rate).
    plot : bool
        If True, produce a quick-look plot (time series + PSD).
    title : str, optional
        Plot title.
    max_freq : float, optional
        Upper frequency limit for plotting [Hz].
    ax : tuple(Axes, Axes), optional
        Provide (ax_time, ax_psd). If None and plot=True, a new figure is created.

    Returns
    -------
    FFTAnalysisResult
        Contains frequency grid, FFT, raw power, and rms^2/Hz PSD.
    """
    x = np.asarray(rate, dtype=float)
    if x.ndim != 1:
        raise ValueError("rate must be a 1D array")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite float")

    n0 = x.size
    mean_rate = float(np.mean(x))

    # Detrend
    if detrend == "none":
        x_d = x.copy()
    elif detrend == "mean":
        x_d = x - mean_rate
    elif detrend == "linear":
        t = np.arange(n0) * dt
        p = np.polyfit(t, x, deg=1)
        trend = np.polyval(p, t)
        x_d = x - trend
    else:
        raise ValueError("detrend must be one of: 'none','mean','linear'")

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

    xw = x_d * w

    # NFFT
    if nfft is None:
        n = n0
    else:
        n = int(nfft)
        if n <= 0:
            raise ValueError("nfft must be a positive integer")
        if n > n0:
            xw = np.pad(xw, (0, n - n0), mode="constant", constant_values=0.0)
        elif n < n0:
            xw = xw[:n]

    # FFT (one-sided)
    X = rfft(xw)
    f = rfftfreq(n, dt)

    # Raw one-sided power
    Praw = np.abs(X) ** 2

    # Window power correction U = mean(w^2)
    # Scale for one-sided PSD in rms^2/Hz:
    # PSD = (2*dt)/(N*U) * |X|^2   (then DC & Nyquist are not doubled)
    if n == n0:
        U = float(np.mean(w ** 2))
    else:
        # If padding/truncation occurred, approximate U by padding/truncating window too
        ww = w
        if n > n0:
            ww = np.pad(ww, (0, n - n0), mode="constant", constant_values=0.0)
        else:
            ww = ww[:n]
        U = float(np.mean(ww ** 2))

    psd_abs = (2.0 * dt) / (n * U) * Praw
    psd_abs[0] *= 0.5
    if n % 2 == 0 and psd_abs.size > 1:
        psd_abs[-1] *= 0.5

    # Convert to fractional rms^2/Hz if requested
    if normalize == "rms2_per_hz":
        psd_frac = psd_abs / (mean_rate ** 2) if mean_rate != 0.0 else psd_abs.copy()
    elif normalize == "raw":
        psd_frac = psd_abs  # still return abs PSD in psd_rms2_per_hz field
    else:
        raise ValueError("normalize must be 'raw' or 'rms2_per_hz'")

    # Plot
    if plot:
        if ax is None:
            fig = plt.figure(figsize=(10,8))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
        else:
            ax1, ax2 = ax

        tplot = np.arange(n0) * dt
        ax1.plot(tplot, x, lw=1)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Rate (counts/s)")
        if title:
            ax1.set_title(title)

        # Skip DC for log-log plotting
        y = psd_frac
        ax2.plot(f[1:], y[1:], lw=1)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel(
            "PSD (fractional rms$^2$/Hz)" if normalize == "rms2_per_hz" else "PSD (rms$^2$/Hz)"
        )
        # ax2.set_xscale("log")
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
        mean_rate=mean_rate,
        dt=dt,
        n=n,
        meta={
            "detrend": detrend,
            "window": window,
            "normalize": normalize,
            "U_window_power": U,
            "nfft": nfft,
        },
    )


def usage_example(seed: int = 7) -> FFTAnalysisResult:
    """
    Demonstration with simulated Poisson counts from a sinusoidally-modulated rate.
    Produces a plot showing the time series and a PSD peak at the injected frequency.
    """
    rng = np.random.default_rng(seed)

    dt = 0.01          # 10 ms bins
    T = 20.0           # seconds
    n = int(T / dt)
    t = np.arange(n) * dt

    mean_rate = 200.0  # counts/s
    frac_amp = 0.2     # 20% modulation
    f0 = 3.0           # Hz injected signal

    rate_true = mean_rate * (1.0 + frac_amp * np.sin(2 * np.pi * f0 * t))

    # Poisson sample -> counts/bin, then convert back to observed rate
    counts = rng.poisson(rate_true * dt)
    rate_obs = counts / dt

    return fft_rate_analysis(
        rate_obs,
        dt,
        detrend="mean",
        window="hann",
        normalize="rms2_per_hz",
        plot=True,
        title=f"FFT demo: injected {f0} Hz sinusoid, mean={mean_rate} c/s",
        max_freq=50.0,
    )


if __name__ == "__main__":
    usage_example()
