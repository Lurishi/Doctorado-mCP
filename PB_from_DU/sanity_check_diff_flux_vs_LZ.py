# -*- coding: utf-8 -*-
r"""
sanity_check_diff_flux_vs_LZ.py
================================

Sanity check de pb_differential_flux.py: integra el flujo DIFERENCIAL

    Phi_from_diff(mchi) = INT dEchi  d^2Phi_chi^s/(dEchi dOmega)(mchi,Echi)

en un grid de masas, y lo compara contra:

  (a) surface_flux(mchi) de pb_surface_flux.py (la version que integra
      analiticamente la ventana plana en Echi a 1) -- chequeo de consistencia
      INTERNA entre los dos modulos.
  (b) PB_surface_flux_eps0_01.csv -- la curva que en reproduce_LZ_PB_flux.py
      ya fue comparada contra la Fig. 1 de LUX-ZEPLIN (arXiv:2412.04854) y
      dio buen acuerdo de forma y normalizacion (ver LZ_PB_flux_reproduction.png
      y el .tex, Sec. 8). Como no tenemos aqui los puntos digitalizados de LZ
      en si, usamos esa curva ya validada como proxy: si Phi_from_diff la
      reproduce, transitivamente tambien reproduce LZ.

Si todo esta bien, las tres curvas deben coincidir: eso confirma que
pb_differential_flux.py (que agrega la dependencia en Echi que no estaba en
el codigo original) es consistente con toda la cadena ya validada contra LZ.
"""
#%%
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

#%%
from pb_surface_flux import surface_flux
from pb_differential_flux import differential_flux

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

EPS = 0.01

# Grilla de calculo (compromiso costo/precision; ~5-6 s por masa).
GRID = dict(Ep_max=1.0e4, n_Ep=24, n_k2=90, n_E0=40, n_cos=56)
N_ECHI = 50
ECHI_MAX = 1.0e4     # GeV, mismo Ep_max que el corte del flujo de protones


def phi_from_diff(mchi, eps=EPS, n_Echi=N_ECHI, **grid):
    r"""INT dEchi d^2Phi/dEchi(mchi,Echi), integrando en un grid log en Echi
    desde el umbral cinematico mchi hasta ECHI_MAX."""
    Echi = np.logspace(np.log10(mchi * 1.001), np.log10(ECHI_MAX), n_Echi)
    Phi_diff = differential_flux(mchi, Echi, eps=eps, **grid)
    return _trapz(Phi_diff, Echi)


if __name__ == "__main__":
    # Curva de referencia ya benchmarkeada contra LZ (ver .tex, Sec. 8/9).
    ref = pd.read_csv("/home/lurishi/Escritorio/Doctorado/Mcp_gráficos_papers/Flujo_PB.csv", comment="#",sep = ';',
                       header=1, names=["mchi", "phi"])

    # Grid de masas a evaluar (mas ralo que el CSV de referencia por costo
    # computacional del pipeline diferencial: ~5-6 s/masa). Cubrimos la
    # meseta, el codo en m_rho/2~385 MeV y el arranque de la caida.
    masses = np.array([0.010, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15,
                        0.20, 0.25, 0.30, 0.331, 0.36, 0.385, 0.41, 0.45,
                        0.50, 0.60])

    print("=== Integrando el flujo diferencial (pb_differential_flux) ===")
    t0 = time.time()
    phi_diff_int = np.zeros_like(masses)
    phi_direct = np.zeros_like(masses)
    for i, m in enumerate(masses):
        phi_diff_int[i] = phi_from_diff(m, **GRID)
        phi_direct[i] = surface_flux(m, eps=EPS, **GRID)
        print(f"  mchi={m*1e3:6.1f} MeV   "
              f"Phi(INT dEchi diff)={phi_diff_int[i]:.3e}   "
              f"Phi(surface_flux)={phi_direct[i]:.3e}   "
              f"cociente={phi_diff_int[i]/phi_direct[i]:.3f}")
    print(f"Tiempo total: {time.time()-t0:.1f} s")

    # -----------------------------------------------------------------
    # Figura: las tres curvas superpuestas + panel de cociente vs CSV ref.
    # -----------------------------------------------------------------
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1]))

    ax.plot(ref["mchi"] * 1e3, ref["phi"], "k--", lw=2,
            label="surface_flux (ya validado vs LZ Fig. 1)")
    ax.plot(masses * 1e3, phi_direct, "bo", ms=5,
            label="surface_flux (puntos de este chequeo)")
    ax.plot(masses * 1e3, phi_diff_int, "r-", lw=2, marker="s", ms=4,
            label=r"$\int dE_\chi\, d^2\Phi/dE_\chi\,d\Omega$ (nuevo modulo)")
    ax.axvline(770 / 2, color="gray", ls=":", lw=1, label=r"$m_\rho/2$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"Total Flux $\Phi_\chi$ [s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")
    ax.set_title(r"Sanity check: flujo diferencial integrado vs LZ, $\epsilon=0.01$")
    ax.set_xlim(10, 700)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # Cociente contra la curva de referencia (interpolada en log-log).
    ref_interp = np.interp(np.log(masses), np.log(ref["mchi"]),
                            np.log(ref["phi"]))
    ratio = phi_diff_int / np.exp(ref_interp)
    axr.axhline(1.0, color="gray", lw=1)
    axr.plot(masses * 1e3, ratio, "r-o", ms=4)
    axr.set_ylim(0.5, 1.5)
    axr.set_xlabel(r"$m_\chi$ [MeV/$c^2$]")
    axr.set_ylabel("cociente\nvs LZ-benchmark")
    axr.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("sanity_check_diff_flux_vs_LZ.png", dpi=150)
    print("\nGuardado: sanity_check_diff_flux_vs_LZ.png")

    print(f"\nCociente [INT dEchi diff] / [surface_flux], min-max: "
          f"{ (phi_diff_int/phi_direct).min():.3f} - "
          f"{ (phi_diff_int/phi_direct).max():.3f}")

# %%
fig = plt.figure(figsize=(7, 6))
plt.plot(masses,ref_interp(masses))
plt.plot(masses * 1e3, phi_direct, "bo", ms=5)

plt.xscale('log')
plt.yscale('log')
plt.savefig("flujo_PB.png", dpi=150)
plt.show()
# %%
