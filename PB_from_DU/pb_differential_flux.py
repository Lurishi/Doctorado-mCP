# -*- coding: utf-8 -*-
r"""
pb_differential_flux.py
========================

Flujo DIFERENCIAL de superficie  d^2 Phi_chi^s / (dEchi dOmega)  por proton
bremsstrahlung (PB), extendiendo pb_surface_flux.py (que solo daba el flujo
integrado en energia).

REFERENCIA
----------
    [Du]  M. Du, R. Fang, Z. Liu, JHEP 08 (2024) 174, arXiv:2211.11469.

IDEA (ver "Proximos pasos" del .tex adjunto)
---------------------------------------------
La distribucion en Echi de una mCP producida en la interaccion es PLANA
entre E_- y E_+ (texto bajo Ec. 3.1 de [Du]):

    dN_chi/dEchi |_{k^2, s, Echi}  =  Theta(Echi-E_-) Theta(E_+-Echi) / (E_+ - E_-)

con, evaluando en el LAB para el foton off-shell de momento (Ek_lab, |k|_lab):

    E_pm = Ek_lab/2  +-  (1/2) |k|_lab sqrt(1 - 4 mchi^2/k^2)

(Ek_lab^2 - |k|_lab^2 = k^2 es invariante, de ahi que sqrt(Ek_lab^2-k^2)=|k|_lab.)

Insertando esto en la Ec. (3.1) integrada (nuestra multiplicity() de
pb_surface_flux.py, que integraba dN_chi/dEchi en Echi y por eso la
distribucion plana desaparecia dando 1), NO se integra en Echi y queda:

    dN_chi/dEchi (Ep, mchi, Echi) =
        (eps^2 e^2 / 6 pi^2) INT_{4mchi^2}^{k2max} dk^2/k^2
            sqrt(1-4mchi^2/k^2)(1+2mchi^2/k^2) |F_V(k^2)|^2  J(k^2, s, Echi)

    J(k^2,s,Echi) = INT dEk0 dcos0  (d^2P/dEk0 dcos0) |F_*(pp-k)|^2
                      * [Theta(E_- <= Echi <= E_+) / (E_+ - E_-)]

(mismo dominio (Ek0,cos0) del CM que en inner_integral() de
pb_surface_flux.py; solo se agrega la ventana/pesos en Echi.)

Y el flujo diferencial de superficie ([Du], Ec. 2.1, con el mismo colapso de
la cascada documentado en pb_surface_flux.py):

    d^2 Phi_chi^s/(dEchi dOmega) (mchi, Echi) =
        INT dEp  Phi_p(Ep)  dN_chi/dEchi (Ep, mchi, Echi)

NOTA DE UNIDADES: Phi_diff tiene unidades cm^-2 s^-1 sr^-1 GeV^-1. Integrando
en Echi se debe recuperar (numericamente) el flujo total de surface_flux().

VECTORIZACION
-------------
Para no pagar el costo de recorrer Echi como bucle externo (redundante:
recorreria de nuevo TODA la cadena k2 x Ep x (Ek0,cos0) por cada Echi), este
modulo vectoriza en Echi como el ULTIMO eje: por cada (Ep, k2) se calcula una
sola vez la grilla (Ek0,cos0) -> (E_-,E_+) y se contrae contra un array de
Echi de una sola vez (broadcasting), igual que ya se hace con el trapecio en
(Ek0,cos0). Esto hace que el costo escale como antes (~ n_k2 * n_Ep grillas
2D) y NO como n_k2*n_Ep*n_Echi.
"""
#%%
import numpy as np
#%%
from pb_splitting_kernel import (
    ALPHA, M_P, F_V, F_star, kernel_CM, Ek0_max, s_from_Ep, cm_to_lab,
)
from pb_surface_flux import (
    E2, PHI_P_NORM, PHI_P_INDEX, proton_flux, build_k2_grid, _trapz,
)

__all__ = [
    "energy_window", "inner_integral_diff", "multiplicity_diff",
    "differential_flux", "check_normalization",
]


# ---------------------------------------------------------------------------
# Ventana cinematica en Echi  (E_-, E_+)  para un foton off-shell (Ek_lab,k^2)
# ---------------------------------------------------------------------------
def energy_window(Ek_lab, kmag_lab, k2, mchi):
    r"""Devuelve (E_minus, E_plus, width) para la distribucion plana en Echi.

        E_pm = Ek_lab/2 +- (1/2) |k|_lab sqrt(1 - 4 mchi^2/k^2)
        width = E_+ - E_- = |k|_lab sqrt(1 - 4 mchi^2/k^2)

    Si 4 mchi^2 > k^2 (por debajo del umbral cinematico gamma*->chi chibar
    para ese k^2 particular) devuelve width=0 (ventana vacia); esto solo
    puede pasar si k2 < 4 mchi^2, que ya deberia estar excluido por los
    limites de integracion en k^2 (k2min = 4 mchi^2), pero se deja el guard
    por robustez numerica en el borde.
    """
    phase = np.clip(1.0 - 4.0 * mchi**2 / k2, 0.0, None)
    width = kmag_lab * np.sqrt(phase)
    Emid = 0.5 * Ek_lab
    half = 0.5 * width
    return Emid - half, Emid + half, width


# ---------------------------------------------------------------------------
# Integral interna J(k^2,s,Echi)  (2D en el CM, vectorizada en Echi)
# ---------------------------------------------------------------------------
def inner_integral_diff(s, Ep, mk, mchi, Echi, n_E0=80, n_cos=120,
                         cos_stretch=True):
    r"""J(k^2,s,Echi) para un array Echi (misma logica que inner_integral()
    de pb_surface_flux.py, pero pesando cada punto (Ek0,cos0) con la ventana
    plana en Echi en vez de integrar/colapsar sobre ella).

    Devuelve un array de la misma forma que Echi.
    """
    E0max = Ek0_max(s, mk)
    Echi = np.asarray(Echi, dtype=float)
    if E0max <= mk:
        return np.zeros_like(Echi)

    E0 = np.linspace(mk, E0max, n_E0)
    if cos_stretch:
        u = np.linspace(0.0, 1.0, n_cos)
        cos0 = -1.0 + 2.0 * u**1.5
    else:
        cos0 = np.linspace(-1.0, 1.0, n_cos)

    E0g, cos0g = np.meshgrid(E0, cos0, indexing="ij")   # (n_E0, n_cos)

    K = kernel_CM(s, E0g, cos0g, mk)

    Ek_lab, cos_lab, kmag_lab, _ = cm_to_lab(E0g, cos0g, s, mk)
    ppz = np.sqrt(max(Ep**2 - M_P**2, 0.0))
    k2 = mk**2
    p2 = M_P**2 + k2 - 2.0 * Ep * Ek_lab + 2.0 * ppz * kmag_lab * cos_lab
    Fstar2 = F_star(p2) ** 2

    base = K * Fstar2                                    # (n_E0, n_cos)
    Emin, Emax, width = energy_window(Ek_lab, kmag_lab, k2, mchi)  # (n_E0,n_cos)

    # inv_width: 1/width donde width>0, si no 0 (ventana degenerada evitada).
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_width = np.where(width > 0, 1.0 / width, 0.0)

    # Broadcasting: Echi -> (n_Echi,1,1); grillas -> (1,n_E0,n_cos).
    Echi_b   = Echi[:, None, None]
    Emin_b   = Emin[None, :, :]
    Emax_b   = Emax[None, :, :]
    inv_w_b  = inv_width[None, :, :]
    base_b   = base[None, :, :]

    inside = (Echi_b >= Emin_b) & (Echi_b <= Emax_b)
    integrand = np.where(inside, base_b * inv_w_b, 0.0)   # (n_Echi, n_E0, n_cos)

    # Trapecios 2D: primero en cos0 (ultimo eje), luego en E0.
    I_cos = _trapz(integrand, cos0, axis=2)               # (n_Echi, n_E0)
    J = _trapz(I_cos, E0, axis=1)                         # (n_Echi,)
    return J


# ---------------------------------------------------------------------------
# dN_chi/dEchi (Ep, mchi, Echi)   [Du, Ec. 3.1, sin integrar en Echi]
# ---------------------------------------------------------------------------
def multiplicity_diff(Ep, mchi, Echi, eps=1.0, n_k2=160, n_E0=80, n_cos=120):
    r"""dN_chi/dEchi(Ep,mchi,Echi) [1/GeV], array de la forma de Echi.

    Igual que multiplicity() de pb_surface_flux.py pero conservando la
    dependencia en Echi (no integra la ventana plana a 1).
    """
    Echi = np.atleast_1d(np.asarray(Echi, dtype=float))
    s = s_from_Ep(Ep)
    sqs = np.sqrt(s)
    mk_max = sqs - 2.0 * M_P
    if mk_max <= 0:
        return np.zeros_like(Echi)
    k2min = 4.0 * mchi**2
    k2max = mk_max**2
    if k2min >= k2max:
        return np.zeros_like(Echi)

    k2grid = build_k2_grid(k2min, k2max, n_k2)
    integ = np.zeros((len(k2grid), len(Echi)))
    for i, k2 in enumerate(k2grid):
        mk = np.sqrt(k2)
        ps = (1.0 / k2) * np.sqrt(max(1.0 - 4.0 * mchi**2 / k2, 0.0)) \
             * (1.0 + 2.0 * mchi**2 / k2)
        FV2 = np.abs(F_V(k2)) ** 2
        J = inner_integral_diff(s, Ep, mk, mchi, Echi, n_E0=n_E0, n_cos=n_cos)
        integ[i, :] = ps * FV2 * J

    dN = (eps**2 * E2 / (6.0 * np.pi**2)) * _trapz(integ, k2grid, axis=0)
    return dN


# ---------------------------------------------------------------------------
# Flujo diferencial de superficie  d^2Phi_chi^s/(dEchi dOmega)  [Du, Ec. 2.1]
# ---------------------------------------------------------------------------
def differential_flux(mchi, Echi, eps=0.01, Ep_max=1.0e4, n_Ep=48,
                       n_k2=160, n_E0=80, n_cos=120, return_curve=False):
    r"""d^2Phi_chi^s/(dEchi dOmega)(mchi,Echi)  [cm^-2 s^-1 sr^-1 GeV^-1].

        = INT dEp  Phi_p(Ep)  dN_chi/dEchi(Ep,mchi,Echi)

    Echi puede ser escalar o array; el umbral en Ep es el mismo que en
    surface_flux() (se necesita sqrt(s) >= 2mp+2mchi para que exista
    k2max >= 4mchi^2).

    return_curve=True devuelve tambien (Ep, dN_chi/dEchi(Ep, Echi)) [shape
    (n_Ep, n_Echi)] para diagnostico/depuracion.
    """
    Echi = np.atleast_1d(np.asarray(Echi, dtype=float))
    s_thr = (2.0 * M_P + 2.0 * mchi) ** 2
    Ep_thr = (s_thr - 2.0 * M_P**2) / (2.0 * M_P)
    if Ep_thr >= Ep_max:
        zero = np.zeros_like(Echi)
        return (zero, None, None) if return_curve else zero

    Ep = np.logspace(np.log10(Ep_thr * 1.0005), np.log10(Ep_max), n_Ep)
    dNvals = np.zeros((n_Ep, len(Echi)))
    for j, E in enumerate(Ep):
        dNvals[j, :] = multiplicity_diff(E, mchi, Echi, eps=eps,
                                          n_k2=n_k2, n_E0=n_E0, n_cos=n_cos)

    integrand = proton_flux(Ep)[:, None] * dNvals      # (n_Ep, n_Echi)
    Phi = _trapz(integrand, Ep, axis=0)                # (n_Echi,)
    if return_curve:
        return Phi, Ep, dNvals
    return Phi


# ---------------------------------------------------------------------------
# Chequeo de consistencia: INT dEchi Phi_diff  debe dar  surface_flux()
# ---------------------------------------------------------------------------
def check_normalization(mchi, eps=0.01, n_Echi=40, **grid_kwargs):
    r"""Compara INT dEchi d^2Phi/dEchi dOmega  contra surface_flux(mchi,eps).

    Util como test de consistencia interna entre este modulo y
    pb_surface_flux.py. Requiere elegir un rango de Echi que cubra
    razonablemente la ventana fisica (Echi de mchi hasta ~ Ep_max).
    """
    from pb_surface_flux import surface_flux

    Phi_total = surface_flux(mchi, eps=eps, **grid_kwargs)

    # Rango de Echi: desde mchi (reposo) hasta un valor generoso.
    Ep_max = grid_kwargs.get("Ep_max", 1.0e4)
    Echi_grid = np.logspace(np.log10(mchi * 1.001), np.log10(Ep_max), n_Echi)
    Phi_diff = differential_flux(mchi, Echi_grid, eps=eps, **grid_kwargs)
    Phi_integrated = _trapz(Phi_diff, Echi_grid)

    print(f"mchi={mchi:.3f} GeV, eps={eps}:")
    print(f"  surface_flux (integrado analiticamente en Echi) = {Phi_total:.4e}")
    print(f"  INT dEchi d^2Phi/dEchi                          = {Phi_integrated:.4e}")
    if Phi_total > 0:
        print(f"  cociente = {Phi_integrated/Phi_total:.4f}")
    return Phi_total, Phi_integrated


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # ------------------------------------------------------------------
    # 1) Chequeo de consistencia (grilla chica para que sea rapido).
    # ------------------------------------------------------------------
    GRID_FAST = dict(Ep_max=1.0e4, n_Ep=20, n_k2=60, n_E0=32, n_cos=48)
    print("=== Chequeo de normalizacion (Phi_diff integrado vs surface_flux) ===")
    check_normalization(0.1, eps=0.01, n_Echi=30, **GRID_FAST)

    # ------------------------------------------------------------------
    # 2) Curvas dPhi/dEchi para varias masas, a eps fijo.
    # ------------------------------------------------------------------
    print("\n=== Calculando dPhi/dEchi vs Echi para varias masas ===")
    masses = [0.03, 0.1, 0.2, 0.35]     # GeV
    eps_fixed = 0.01
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in masses:
        Echi_grid = np.logspace(np.log10(m * 1.001), np.log10(500.0), 60)
        Phi = differential_flux(m, Echi_grid, eps=eps_fixed, **GRID_FAST)
        ax.plot(Echi_grid, Phi, lw=1.8, label=fr"$m_\chi={m*1e3:.0f}$ MeV")
        print(f"  mchi={m*1e3:.0f} MeV: listo")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E_\chi$ [GeV]")
    ax.set_ylabel(r"$d^2\Phi^s_\chi/(dE_\chi\,d\Omega)$ [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]")
    ax.set_title(fr"Flujo diferencial PB, $\epsilon={eps_fixed}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dPhi_dEchi_vs_mass.png", dpi=140)
    print("Guardado: dPhi_dEchi_vs_mass.png")

    # ------------------------------------------------------------------
    # 3) Curvas dPhi/dEchi para varios epsilon, a masa fija.
    # ------------------------------------------------------------------
    print("\n=== Calculando dPhi/dEchi vs Echi para varios epsilon ===")
    LZ = pd.read_csv('/home/lurishi/Escritorio/Doctorado/PB_from_DU/PB_cos_1.csv',sep = ';')

    E_lz = np.sort(LZ[LZ.columns[0]].values)

    Phi_lz = np.sort(LZ[LZ.columns[1]].values)[::-1]
    
    m_fixed = 0.1
    eps_list = [ 0.01]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(E_lz, Phi_lz, "k--", lw=2, label="PB (LZ, arXiv:2412.04854)")
    Echi_grid = np.logspace(np.log10(m_fixed * 1.001), np.log10(500.0), 60)
    for e in eps_list:
        Phi = differential_flux(m_fixed, Echi_grid, eps=e, **GRID_FAST)
        ax.plot(Echi_grid, Phi, lw=1.8, label=fr"$\epsilon={e}$")
        print(f"  eps={e}: listo")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E_\chi$ [GeV]")
    ax.set_ylabel(r"$d^2\Phi^s_\chi/(dE_\chi\,d\Omega)$ [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]")
    ax.set_title(fr"Flujo diferencial PB, $m_\chi={m_fixed*1e3:.0f}$ MeV")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig("dPhi_dEchi_vs_eps.png", dpi=140)
    print("Guardado: dPhi_dEchi_vs_eps.png")

# %%
