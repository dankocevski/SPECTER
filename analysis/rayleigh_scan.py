from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple
from scipy.signal import correlate, correlation_lags, detrend
from typing import Iterable, Optional, Tuple, Sequence, Dict
from scipy.stats import norm

try:
    from scipy.signal import find_peaks
except Exception as e:  # pragma: no cover
    find_peaks = None

color = "#36498D"

def rayleigh_scan(events: np.ndarray, fmin: float, fmax: float, N: int = 2000):
    freqs = np.geomspace(max(fmin, 1e-6), fmax, N)
    Z = np.empty_like(freqs)
    for j, f in enumerate(freqs):
        phi = 2 * np.pi * f * events
        C = np.cos(phi).sum(); S = np.sin(phi).sum()
        Z[j] = (2.0 / len(events)) * (C * C + S * S)
    return freqs, Z


def _detect_rayleigh_peaks(freqs: np.ndarray, Z: np.ndarray, top_k: int = 3):
    """Find top candidate peaks in a Rayleigh scan by local maxima and height."""
    if find_peaks is None:
        # fallback: take the top_k by value (excluding edges)
        idx = np.argsort(Z[1:-1])[::-1][:top_k] + 1
        return [(freqs[i], Z[i]) for i in idx]
    pk_idx, _ = find_peaks(Z)
    if pk_idx.size == 0:
        return []
    # Sort by Z height descending
    order = np.argsort(Z[pk_idx])[::-1]
    pk_idx = pk_idx[order][:top_k]
    return [(freqs[i], Z[i]) for i in pk_idx]


def _bin_for_display(events: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    t0 = float(events.min())
    t1 = float(events.max())
    nb = max(10, int(np.ceil((t1 - t0) / dt)))
    edges = np.linspace(t0, t1, nb + 1)
    counts, _ = np.histogram(events, bins=edges)
    t = 0.5 * (edges[:-1] + edges[1:])
    return t, counts

# -----------------------------
# Plotting helpers
# -----------------------------

def plot_lightcurve(t: np.ndarray, x: np.ndarray, ax: Optional[plt.Axes] = None, t_ref: Optional[float] = None, label: Optional[str] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    tplot = t - (t_ref if t_ref is not None else 0.0)
    ax.plot(tplot, x, lw=1.0, color=color)
    ax.set_xlabel("Time (s)" + (" − t_ref" if t_ref is not None else ""))
    ax.set_ylabel("Counts")
    if label:
        ax.legend([label])
    ax.grid(True, alpha=0.3)
    return ax


def plot_rayleigh_scan(freqs: np.ndarray, Z: np.ndarray, N_trials: Optional[int] = None, alpha_levels: Sequence[float] = (1e-2, 1e-3), ax: Optional[plt.Axes] = None, title: Optional[str] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freqs, Z, lw=1.0, color=color)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Rayleigh Z₁²')

    if N_trials is not None:
        for a, ls in zip(alpha_levels, [':', '--']):
            alpha_single = 1.0 - (1.0 - a) ** (1.0 / max(N_trials, 1))
            zthr = -np.log(1.0 - alpha_single)
            ax.axhline(zthr, ls=ls, alpha=0.6, label=f"{a*100:.2g}% global FAP")

    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best')
    return ax

def plot_acf(x: np.ndarray, dt: float, max_lag: Optional[float] = None, ax: Optional[plt.Axes] = None, title: Optional[str] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    x0 = np.asarray(x) - np.mean(x)
    c = correlate(x0, x0, mode='full')
    lags = correlation_lags(len(x0), len(x0), mode='full') * dt
    c /= c.max() if c.max() != 0 else 1

    if max_lag is not None:
        m = np.abs(lags) <= max_lag
        lags, c = lags[m], c[m]

    ax.plot(lags, c, lw=1.0, color=color)
    ax.set_xlabel('Lag (s)')
    ax.set_ylabel('ACF')
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_lorentzian_zoom(freqs: np.ndarray,
                         power: np.ndarray,
                         nu0: float,
                         width: float,
                         continuum_level: Optional[float] = None,
                         window_halfwidth: float = 3.0,
                         ax: Optional[plt.Axes] = None,
                         title: Optional[str] = None) -> plt.Axes:
    """Zoom in around a candidate QPO and overlay a Lorentzian fit.

    The Lorentzian drawn is \n
        L(ν) = A * (0.5*Γ)^2 / ((ν-ν0)^2 + (0.5*Γ)^2) + C

    with Γ = width and C = continuum_level (if provided). Amplitude A is
    chosen to match the peak power at ν0.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    f_lo = max(1e-12, nu0 - window_halfwidth * width)
    f_hi = nu0 + window_halfwidth * width
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    f = freqs[mask]
    P = power[mask]

    ax.plot(f, P, drawstyle='steps-mid', lw=1.0)

    # Fit amplitude at the closest bin to nu0
    if len(f) > 0:
        k = np.argmin(np.abs(f - nu0))
        Pk = P[k]
        C = continuum_level if continuum_level is not None else np.median(P)
        A = max(Pk - C, 0.0)
        nu = np.linspace(f_lo, f_hi, 512)
        L = A * (0.5 * width) ** 2 / ((nu - nu0) ** 2 + (0.5 * width) ** 2) + C
        ax.plot(nu, L, lw=2.0, label='Lorentzian fit')
        if continuum_level is not None:
            ax.axhline(C, ls='--', alpha=0.6, label='Continuum')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (Leahy)')
    if title is None:
        title = f"Zoom near ν₀≈{nu0:.3g} Hz"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return ax

# -----------------------------
# Run the search
# -----------------------------

def run(
    events: np.ndarray,
    *,
    fmin: float = 0.5,
    fmax: float = 1000.0,
    nfreq: int = 2000,
    dt_plot: float = 0.064,
    maxlag: Optional[float] = None,
    out: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, Any]]:
    """Run an unbinned (Rayleigh) QPO search and generate a summary figure (wavelet as panel 4)."""

    events = np.asarray(events, dtype=float)
    if events.ndim != 1 or events.size < 10:
        raise ValueError("'events' must be a 1D array with at least ~10 entries")

    # Ensure sorted for nicer plotting and numerical stability
    events = np.sort(events)

    # --- Rayleigh scan on unbinned events ---
    freqs, Z = rayleigh_scan(events, fmin, fmax, N=nfreq)
    N_trials = len(freqs)

    # Global-FAP conversion for exponential tail: p_single = exp(-Z)
    def p_global_from_Z(z):
        p_single = np.exp(-max(z, 0.0))
        return 1.0 - (1.0 - p_single) ** N_trials
    
    def sigma_from_p(p):
        return norm.isf(p / 2.0)  # two-sided

    # Peak picking
    raw_peaks = _detect_rayleigh_peaks(freqs, Z, top_k=3)
    peak_dicts = [
        {"nu0": float(round(nu0, 3)),
          "Z": float(round(z)), 
          "p_global": float(p_global_from_Z(z)), 
          "sigma_global": float(round(sigma_from_p(p_global_from_Z(z)), 3))}
        for (nu0, z) in raw_peaks
    ]

    # --- Figure layout: 2x2 ---
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax_top = fig.add_subplot(gs[0, :])     # top panel spans both columns
    ax_left = fig.add_subplot(gs[1, 0])    # bottom-left
    ax_right = fig.add_subplot(gs[1, 1])   # bottom-right
    # axes = np.array([ax_top, ax_left, ax_right], dtype=object)
    # axes = axes.ravel()

    # Panel 1: binned light curve (for visualization only)
    t_lc, x_lc = _bin_for_display(events, dt_plot)
    plot_lightcurve(t_lc, x_lc, ax=ax_top, label=f"dt={dt_plot:.3g}s")

    # Panel 2: Rayleigh scan with global-FAP thresholds
    plot_rayleigh_scan(freqs, Z, N_trials=N_trials, ax=ax_left, title="Rayleigh Z1^2 scan")

    # Panel 3: ACF of binned LC (optional maxlag)
    try:
        plot_acf(x_lc, dt_plot, max_lag=maxlag, ax=ax_right, title="Autocorrelation (binned)")
    except Exception:
        ax_right.text(0.5, 0.5, "ACF unavailable", ha="center", va="center")
        ax_right.set_axis_off()


    fig.suptitle("Unbinned QPO search summary (events)", y=0.98)
    fig.tight_layout()

    outpaths: Dict[str, str] = {}
    if out:
        for ext in ("png", "pdf"):
            path = f"{out}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            outpaths[ext] = path

    results: Dict[str, Any] = {
        "freqs": freqs,
        "Z": Z,
        "peaks": peak_dicts,
        "N_trials": N_trials,
        "fmin": fmin,
        "fmax": fmax,
    }

    print("Top candidate peaks:")
    for pk in results["peaks"]:
        print(pk)

    plt.show()

    return results


def test(nu0=5.0, frac_amp=0.15, qpo_width=0.5, t_total=200.0, rate_mean=200.0):
    """Test the Rayleigh QPO search on simulated event data with a weak QPO.""" 
    
    # Random seed for reproducibility
    rng = np.random.default_rng(42)

    # Simulation parameters
    #t_total = 200.0        # total duration in seconds
    #rate_mean = 200.0      # mean count rate (counts per second)
    #nu0 = 5.0              # QPO frequency (Hz)
    #frac_amp = 0.15        # fractional modulation amplitude
    #qpo_width = 0.5        # fractional width (controls phase diffusion)

    # Simulate Poisson process with sinusoidal rate modulation
    t = np.linspace(0, t_total, int(rate_mean * t_total))
    # sinusoidal modulation around mean
    inst_rate = rate_mean * (1 + frac_amp * np.sin(2*np.pi*nu0*t))
    # turn rate curve into cumulative probability
    cdf = np.cumsum(inst_rate)
    cdf /= cdf[-1]
    # sample event times according to non-uniform rate
    events = np.interp(rng.random(int(rate_mean * t_total * (1 - frac_amp**2))), cdf, t)

    # Optionally add phase wander to broaden the peak
    events += rng.normal(scale=qpo_width / nu0 / (2*np.pi), size=events.size)

    # Quick look
    results = run(events, fmin=0.5, fmax=50, nfreq=3000, dt_plot=0.064)
    print("Top candidate peaks:")
    for pk in results["peaks"]:
        print(pk)

    plt.show()