import numpy as np
import matplotlib.pyplot as plt
import pywt

def wavelet_cwt_counts(counts: np.ndarray,
                       dt: float,
                       wavelet: str = "morl",
                       fmin: float | None = None,
                       fmax: float | None = None,
                       voices_per_octave: int = 12,
                       standardize: bool = True):
    """
    Continuous wavelet transform (CWT) of binned counts versus time (fixed dt).

    Parameters
    ----------
    counts : array-like
        Counts per bin (evenly spaced).
    dt : float
        Bin width in seconds.
    wavelet : str
        PyWavelets wavelet name; 'morl' (Morlet) is standard for timing.
    fmin, fmax : float or None
        Frequency band (Hz). If None, fmin defaults to 1/T and fmax to Nyquist (0.5/dt).
    voices_per_octave : int
        Frequency density: more voices -> finer scale sampling.
    standardize : bool
        If True, demean and divide by std before CWT (recommended).

    Returns
    -------
    t : np.ndarray
        Time array at bin centers (s).
    freqs : np.ndarray
        Wavelet pseudo-frequencies (Hz), ascending.
    power : np.ndarray, shape (n_freqs, n_times)
        Wavelet power |W|^2.
    scales : np.ndarray
        Scales used for the CWT (same order as freqs).
    """
    x = np.asarray(counts, dtype=float)
    N = x.size
    T = N * dt
    t = (np.arange(N) + 0.5) * dt  # bin centroids

    if standardize:
        x = x - np.mean(x)
        sx = np.std(x)
        if sx > 0:
            x = x / sx

    # Frequency band
    if fmax is None:
        fmax = 0.5 / dt  # Nyquist
    if fmin is None:
        fmin = 1.0 / T   # about one cycle over full duration

    if fmin <= 0 or fmax <= 0 or fmin >= fmax:
        raise ValueError("Require 0 < fmin < fmax; check dt and array length.")

    # Build logarithmically spaced frequencies (ascending)
    n_octaves = np.log2(fmax / fmin)
    n_voices = max(int(np.ceil(n_octaves * voices_per_octave)) + 1, 2)
    freqs = fmin * (2.0 ** (np.arange(n_voices) / voices_per_octave))

    # Convert frequencies to CWT scales using wavelet's center frequency.
    # PyWavelets: f = fc / (scale * dt)  => scale = fc / (f * dt)
    fc = pywt.central_frequency(wavelet)
    scales = fc / (freqs * dt)

    # Compute CWT (FFT-based for speed)
    coef, used_scales = pywt.cwt(x, scales, wavelet, sampling_period=dt, method="fft")
    power = np.abs(coef) ** 2  # shape: (n_scales, N)

    # Ensure scales/freqs ascending order corresponds to rows in 'power'
    # (With our construction, it already does: small->large scales map high->low freq,
    # but freqs are ascending, so rows align with freqs we built.)
    return t, freqs, power, scales


def plot_wavelet(t: np.ndarray,
                 freqs: np.ndarray,
                 power: np.ndarray,
                 ax: plt.Axes | None = None,
                 log_power: bool = True,
                 vmin: float | None = None,
                 vmax: float | None = None,
                 draw_coi: bool = True,
                 dt: float | None = None,
                 wavelet: str = "morl",
                 title: str | None = None) -> plt.Axes:
    """
    Plot wavelet power as a time–frequency scalogram with frequency (Hz) on y-axis.

    Parameters
    ----------
    t, freqs, power : arrays
        Outputs of wavelet_cwt_counts. freqs ascending (Hz), power shape=(n_freqs, n_times).
    log_power : bool
        If True, plot log10(power); else linear.
    draw_coi : bool
        If True, overlay an approximate cone of influence (Morlet).
    dt : float or None
        Sampling interval; required if draw_coi=True to compute COI.
    wavelet : str
        Wavelet name for COI constants ('morl' assumed).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    P = power
    Z = np.log10(P + 1e-16) if log_power else P
    y = freqs
    x = t

    # Set color scaling
    if vmin is None:
        vmin = np.nanpercentile(Z, 5)
    if vmax is None:
        vmax = np.nanpercentile(Z, 99)

    im = ax.imshow(Z,
                   extent=[x[0], x[-1], y[0], y[-1]],
                   origin="lower",
                   aspect="auto",
                   interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("log10 Wavelet Power" if log_power else "Wavelet Power")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)
    ax.grid(False)

    # Optional: Cone of influence (approximate) for Morlet
    if draw_coi:
        if dt is None:
            raise ValueError("dt must be provided for COI overlay.")
        # Torrence & Compo (1998): e-folding time for Morlet ≈ sqrt(2) * scale
        # Convert to a frequency boundary: f_coi(t) = fc / (dt * coi_time(t))
        fc = pywt.central_frequency(wavelet)
        # Build time-dependent COI half-width in seconds
        # Distance to nearest edge:
        t0, t1 = x[0], x[-1]
        # distance from each time to the closest edge
        dist = np.minimum(x - t0, t1 - x)
        # convert time half-width to "scale" half-width (s -> samples)
        coi_time = np.sqrt(2.0) * dist
        # avoid division by zero at edges
        coi_time = np.maximum(coi_time, 1e-12)
        f_coi = fc / (dt * coi_time)

        # Plot as a curve on top (clip to y-range)
        ax.plot(x, np.minimum(f_coi, y[-1]), color="white", lw=1.0, alpha=0.9, label="COI (approx)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    return ax
    # plt.show()

# -------------------------
# Minimal demo / smoke test
# -------------------------

def simulate_sinusoidal_counts(duration=64.0, dt=0.01,
                                mean_rate=2000.0, mod_freq=5.0, frac_amp=0.1, rng=None):
    if rng is None: rng = np.random.default_rng(123)
    t = np.arange(0, duration, dt)
    lam = mean_rate * (1.0 + frac_amp * np.sin(2*np.pi*mod_freq*t))  # cps
    counts = rng.poisson(lam * dt)
    return t, counts

def test():

    # Simulate a sinusoidally modulated Poisson counting process
    rng = np.random.default_rng(123)
    duration = 32.0   # s
    dt = 0.01         # s  -> Nyquist = 50 Hz
    t = (np.arange(int(duration/dt)) + 0.5) * dt
    mean_rate = 1500.0      # counts/s
    mod_freq = 5.0          # Hz
    frac_amp = 0.2         # 12% fractional modulation
    lam = mean_rate * (1.0 + frac_amp * np.sin(2*np.pi*mod_freq*t))
    counts = rng.poisson(lam * dt)

    # Run CWT over a sensible band
    t_, freqs, power, scales = wavelet_cwt_counts(counts, dt, wavelet="morl",
                                                fmin=1, fmax=20.0, voices_per_octave=16)

    # 1) FFT peak (ground truth)
    ff, PP = np.fft.rfftfreq(t.size, dt), (np.abs(np.fft.rfft(t - t.mean()))**2)
    f_fft = ff[np.argmax(PP[1:])+1]

    # 2) CWT “peak” row
    row = np.argmax(power.mean(axis=1))  # time-averaged power per frequency row
    f_cwt = freqs[row]

    print(f"FFT peak ~ {f_fft:.3f} Hz   |   CWT peak label ~ {f_cwt:.3f} Hz")
    print(f"dt used in CWT = {dt}")

    # Plot
    ax = plot_wavelet(t_, freqs, power, dt=dt, title="Wavelet power (Morlet CWT)")
    # Optional: mark the injected frequency
    ax.axhline(mod_freq, color="C3", lw=1.0, ls="--", alpha=0.8)
    plt.tight_layout()
    plt.show()

    return freqs, power