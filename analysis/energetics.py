import numpy as np
from scipy.integrate import quad

import astropy.units as u
from astropy.cosmology import Planck18 as cosmo  # commonly used in modern GRB papers


# ---------------------------
# Spectral models N(E) in ph / (s cm^2 keV) up to an overall normalization A
# We only need shapes for k-correction (A cancels in the ratio).
# ---------------------------

def _band_N(E_keV, alpha, beta, Epk_keV):
    """
    Band function photon spectrum shape N(E) (normalization cancels in k).
    Epk is the nuFnu peak energy (keV), where nuFnu peaks at Epk.
    """
    E0 = Epk_keV / (2.0 + alpha)  # Band parameterization: Epk = (2+alpha)E0
    Eb = (alpha - beta) * E0

    if np.isscalar(E_keV):
        E = float(E_keV)
        if E < Eb:
            return (E / 100.0) ** alpha * np.exp(-E / E0)
        else:
            return ((Eb / 100.0) ** (alpha - beta) *
                    np.exp(beta - alpha) *
                    (E / 100.0) ** beta)
    else:
        E = np.asarray(E_keV, dtype=float)
        out = np.empty_like(E)
        m = E < Eb
        out[m] = (E[m] / 100.0) ** alpha * np.exp(-E[m] / E0)
        out[~m] = ((Eb / 100.0) ** (alpha - beta) *
                   np.exp(beta - alpha) *
                   (E[~m] / 100.0) ** beta)
        return out


def _cpl_N(E_keV, alpha, Epk_keV):
    """
    Cutoff power law photon spectrum shape N(E) (normalization cancels in k).
    Epk is the nuFnu peak energy (keV): Epk = (2+alpha)E0.
    """
    E0 = Epk_keV / (2.0 + alpha)
    E = np.asarray(E_keV, dtype=float)
    return (E / 100.0) ** alpha * np.exp(-E / E0)


def _pl_N(E_keV, gamma):
    """
    Simple power law photon spectrum shape N(E) ~ E^(-gamma) (normalization cancels in k).
    Note: GRB papers sometimes use photon index 'Gamma' with N(E) ∝ E^(-Gamma).
    """
    E = np.asarray(E_keV, dtype=float)
    return (E / 100.0) ** (-gamma)


def _energy_flux_integral(model_N, Emin_keV, Emax_keV, **params):
    """
    Compute integral of E*N(E) dE over [Emin, Emax] (shape-only; normalization cancels).
    """
    def integrand(E):
        return E * model_N(E, **params)

    val, _ = quad(integrand, Emin_keV, Emax_keV, epsabs=0, epsrel=1e-6, limit=200)
    return val


def k_correction_energy_flux(
    z,
    model="band",
    Eobs_min_keV=10.0,
    Eobs_max_keV=1000.0,
    Erest_min_keV=1.0,
    Erest_max_keV=1.0e4,
    **spec_params
):
    """
    k-correction for converting observed-band *energy flux* to a fixed rest-frame band.

    k = ∫_{Erest/(1+z)} E N(E) dE  /  ∫_{Eobs} E N(E) dE
    """
    model = model.lower()
    if model == "band":
        model_N = _band_N
        required = ("alpha", "beta", "Epk_keV")
    elif model in ("cpl", "cutoffpl", "cutoff_powerlaw"):
        model_N = _cpl_N
        required = ("alpha", "Epk_keV")
    elif model in ("pl", "powerlaw"):
        model_N = _pl_N
        required = ("gamma",)
    else:
        raise ValueError(f"Unknown model='{model}'. Use 'band', 'cpl', or 'pl'.")

    for r in required:
        if r not in spec_params:
            raise ValueError(f"Missing required parameter '{r}' for model '{model}'.")

    Emin_bolo = Erest_min_keV / (1.0 + z)
    Emax_bolo = Erest_max_keV / (1.0 + z)

    num = _energy_flux_integral(model_N, Emin_bolo, Emax_bolo, **spec_params)
    den = _energy_flux_integral(model_N, Eobs_min_keV, Eobs_max_keV, **spec_params)

    return num / den


def flux_to_Liso(
    flux_erg_cm2_s,
    z,
    model="band",
    Eobs_min_keV=10.0,
    Eobs_max_keV=1000.0,
    Erest_min_keV=1.0,
    Erest_max_keV=1.0e4,
    k_correct = False,
    **spec_params,

):
    """
    Isotropic-equivalent luminosity Liso in rest-frame [Erest_min, Erest_max] keV.

    Inputs
    ------
    flux_erg_cm2_s : float or Quantity
        Observed *energy flux* in the instrument band [Eobs_min, Eobs_max] (erg/cm^2/s)
    z : float
        Redshift
    model : 'band' | 'cpl' | 'pl'
        Spectral shape used for k-correction
    spec_params :
        Band: alpha, beta, Epk_keV
        CPL : alpha, Epk_keV
        PL  : gamma  (photon index in N(E) ∝ E^{-gamma})

    Returns
    -------
    Liso : Quantity
        erg/s
    """
    if not hasattr(flux_erg_cm2_s, "unit"):
        F = flux_erg_cm2_s * (u.erg / u.cm**2 / u.s)
    else:
        F = flux_erg_cm2_s.to(u.erg / u.cm**2 / u.s)

    if k_correct:
        k = k_correction_energy_flux(
            z,
            model=model,
            Eobs_min_keV=Eobs_min_keV,
            Eobs_max_keV=Eobs_max_keV,
            Erest_min_keV=Erest_min_keV,
            Erest_max_keV=Erest_max_keV,
            **spec_params
        )
    else:
        k = 1.0

    dL = cosmo.luminosity_distance(z).to(u.cm)
    Liso = 4.0 * np.pi * dL**2 * F * k
    return Liso.to(u.erg / u.s)


def fluence_to_Eiso(
    fluence_erg_cm2,
    z,
    model="band",
    Eobs_min_keV=10.0,
    Eobs_max_keV=1000.0,
    Erest_min_keV=1.0,
    Erest_max_keV=1.0e4,
    k_correct = False,
    **spec_params
):
    """
    Isotropic-equivalent energy Eiso in rest-frame [Erest_min, Erest_max] keV.

    Parameters
    ----------
    fluence_erg_cm2 : float or Quantity
        Observed *energy fluence* in instrument band [Eobs_min, Eobs_max] (erg/cm^2)
    z : float
        Redshift
    model : 'band' | 'cpl' | 'pl'
        Spectral shape used for k-correction
    spec_params :
        Band: alpha, beta, Epk_keV
        CPL : alpha, Epk_keV
        PL  : gamma  (photon index in N(E) ∝ E^{-gamma})

    Returns
    -------
    Eiso : Quantity
        erg
    """
    # Ensure fluence has units
    if not hasattr(fluence_erg_cm2, "unit"):
        S = fluence_erg_cm2 * (u.erg / u.cm**2)
    else:
        S = fluence_erg_cm2.to(u.erg / u.cm**2)

    # Same k-correction used for energy flux (ratio of ∫ E N(E) dE)
    if k_correct:
        k = k_correction_energy_flux(
            z,
            model=model,
            Eobs_min_keV=Eobs_min_keV,
            Eobs_max_keV=Eobs_max_keV,
            Erest_min_keV=Erest_min_keV,
            Erest_max_keV=Erest_max_keV,
            **spec_params
        )
    else:
        k = 1.0

    dL = cosmo.luminosity_distance(z).to(u.cm)
    Eiso = 4.0 * np.pi * dL**2 * S * k / (1.0 + z)
    return Eiso.to(u.erg)


def liso_example():

    F = 2.0e-6  # erg/cm^2/s in 10-1000 keV
    z = 0.42

    Liso = flux_to_Liso(
        F, z,
        model="band",
        Eobs_min_keV=10.0, Eobs_max_keV=1000.0,
        Erest_min_keV=1.0, Erest_max_keV=1.0e4,
        alpha=-1.05, beta=-3.18, Epk_keV=999.0
    )
    print(Liso)

def eiso_example():

    S = 3.2e-5  # erg/cm^2 in 10–1000 keV
    z = 1.2

    Eiso = fluence_to_Eiso(
        S, z,
        model="band",
        Eobs_min_keV=10.0, Eobs_max_keV=1000.0,
        Erest_min_keV=1.0, Erest_max_keV=1.0e4,
        alpha=-1.0, beta=-2.3, Epk_keV=300.0
    )

    print(Eiso)