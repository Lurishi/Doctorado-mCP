# -*- coding: utf-8 -*-
r"""
pb_surface_flux_integrand.py
===========================

Versión de `pb_surface_flux.py` que devuelve el integrando diferencial

dPhi_chi^s/dEp = Phi_p(Ep) * N_chi(Ep, mchi)

en lugar de integrar sobre Ep.

"""
#%%
import numpy as np
#%%

from pb_splitting_kernel import (
    ALPHA, M_P, F_V, F_star, kernel_CM, Ek0_max,
    s_from_Ep, cm_to_lab,
)

# trapz compatible con numpy < 2.0 (np.trapz) y >= 2.0 (np.trapezoid).
# OJO: en numpy 2.x np.trapz fue ELIMINADO (no solo deprecado), por eso no
# se puede usar getattr(np,'trapezoid',np.trapz) (evaluaria np.trapz).
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid        # numpy >= 2.0
else:
    _trapz = np.trapz            # numpy < 2.0 (entorno de Pablo)

E2 = 4.0 * np.pi * ALPHA         # e^2 = 4 pi alpha  (aparece en Du 3.1)

# ---------------------------------------------------------------------------
# Flujo de protones en el tope de la atmosfera  [Du, Ec. (2.2)]
# ---------------------------------------------------------------------------
PHI_P_NORM = 0.74 * 1.8e4 / 1.0e4     # = 1.332
PHI_P_INDEX = -2.7


def proton_flux(Ep):
    r"""Flujo de protones en el tope [cm^-2 s^-1 sr^-1 GeV^-1]  [Du, Ec. 2.2]."""
    return PHI_P_NORM * np.asarray(Ep, dtype=float) ** PHI_P_INDEX


# ---------------------------------------------------------------------------
# Integral interna I(k^2, s, Ep)  (2D en el CM)  con F_* (off-shell)
# ---------------------------------------------------------------------------
def inner_integral(s, Ep, mk, n_E0=80, n_cos=120, cos_stretch=True):
    r"""I(k^2,s) = INT dEk0 dcos0 (d^2P/dEk0 dcos0) |F_*(pp-k)|^2  [adimensional]."""
    E0max = Ek0_max(s, mk)
    if E0max <= mk:
        return 0.0

    E0 = np.linspace(mk, E0max, n_E0)

    if cos_stretch:
        u = np.linspace(0.0, 1.0, n_cos)
        cos0 = -1.0 + 2.0 * u**1.5
    else:
        cos0 = np.linspace(-1.0, 1.0, n_cos)

    E0g, cos0g = np.meshgrid(E0, cos0, indexing="ij")

    K = kernel_CM(s, E0g, cos0g, mk)
    Ek_lab, cos_lab, kmag_lab, _ = cm_to_lab(E0g, cos0g, s, mk)
    ppz = np.sqrt(max(Ep**2 - M_P**2, 0.0))
    p2 = M_P**2 + mk**2 - 2.0 * Ep * Ek_lab + 2.0 * ppz * kmag_lab * cos_lab
    Fstar2 = F_star(p2) ** 2

    integrand = K * Fstar2
    I_cos = _trapz(integrand, cos0, axis=1)
    return _trapz(I_cos, E0)


# ---------------------------------------------------------------------------
# Grilla en k^2 con densidad extra en la resonancia rho/omega
# ---------------------------------------------------------------------------
def build_k2_grid(k2min, k2max, n=160):
    r"""Grilla en k^2 con puntos extra alrededor de m_rho^2 ~ 0.593 GeV^2."""
    base = np.linspace(k2min, k2max, n)
    m_rho2 = 0.77**2
    if k2min < m_rho2 < k2max:
        lo = max(k2min, m_rho2 - 0.15)
        hi = min(k2max, m_rho2 + 0.15)
        extra = np.linspace(lo, hi, n // 2)
        base = np.unique(np.concatenate([base, extra]))
    return base


# ---------------------------------------------------------------------------
# Multiplicidad de mCP por interaccion  N_chi(Ep, mchi)   [Du, Ec. 3.1]
# ---------------------------------------------------------------------------
def multiplicity(Ep, mchi, eps=1.0, n_k2=160, n_E0=80, n_cos=120):
    r"""N_chi por interaccion proton-aire (adimensional), Ec. (3.1) integrada."""
    s = s_from_Ep(Ep)
    sqs = np.sqrt(s)
    mk_max = sqs - 2.0 * M_P
    if mk_max <= 0:
        return 0.0
    k2min = 4.0 * mchi**2
    k2max = mk_max**2
    if k2min >= k2max:
        return 0.0

    k2grid = build_k2_grid(k2min, k2max, n_k2)
    integ = np.zeros_like(k2grid)
    for i, k2 in enumerate(k2grid):
        mk = np.sqrt(k2)
        ps = (1.0 / k2) * np.sqrt(max(1.0 - 4.0*mchi**2/k2, 0.0)) \
             * (1.0 + 2.0*mchi**2/k2)
        FV2 = np.abs(F_V(k2))**2
        I = inner_integral(s, Ep, mk, n_E0=n_E0, n_cos=n_cos)
        integ[i] = ps * FV2 * I

    N = (eps**2 * E2 / (6.0 * np.pi**2)) * _trapz(integ, k2grid)
    return N


# ---------------------------------------------------------------------------
# Flujo de superficie integrado en energia  Phi_chi^s(mchi)   [Du, Ec. 2.1]
# ---------------------------------------------------------------------------
def surface_flux(mchi, eps=0.01, Ep_max=1.0e4, n_Ep=48,
                 n_k2=160, n_E0=80, n_cos=120, return_curve=False):
    r"""Devuelve el integrando dPhi_chi^s/dEp en lugar de la integral total.

    En esta versión la salida es el arreglo
        integrand = Phi_p(Ep) * N_chi(Ep, mchi)
    sobre la grilla en Ep. Si return_curve=True, devuelve tambien
    (Ep, N_chi(Ep)) como diagnostico.
    """
    s_thr = (2.0 * M_P + 2.0 * mchi)**2
    Ep_thr = (s_thr - 2.0 * M_P**2) / (2.0 * M_P)
    if Ep_thr >= Ep_max:
        return (np.array([]), None, None) if return_curve else np.array([])

    Ep = np.logspace(np.log10(Ep_thr * 1.0005), np.log10(Ep_max), n_Ep)
    Nvals = np.array([
        multiplicity(E, mchi, eps=eps, n_k2=n_k2, n_E0=n_E0, n_cos=n_cos)
        for E in Ep
    ])
    integrand = proton_flux(Ep) * Nvals

    if return_curve:
        return integrand, Ep, Nvals
    return integrand


def surface_flux_integrand(mchi, eps=0.01, Ep_max=1.0e4, n_Ep=48,
                          n_k2=160, n_E0=80, n_cos=120):
    """Alias claro para obtener solo el integrando diferencial."""
    return surface_flux(mchi, eps=eps, Ep_max=Ep_max, n_Ep=n_Ep,
                       n_k2=n_k2, n_E0=n_E0, n_cos=n_cos, return_curve=False)


if __name__ == "__main__":
    for mchi in (0.05, 0.1, 0.3):
        Ep, integrand, Nvals = surface_flux(mchi, eps=0.01, n_Ep=20, return_curve=True)
        print(f"mchi={mchi:.3f} GeV:  integrand[0:3] = {integrand[:3]}")
        print(f"mchi={mchi:.3f} GeV:  Ep[0:3] = {Ep[:3]}")

# %%
