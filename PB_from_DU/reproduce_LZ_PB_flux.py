# -*- coding: utf-8 -*-
r"""
reproduce_LZ_PB_flux.py
=======================

Sanity check global: reproduce la curva de flujo PB del panel superior de
la Fig. 1 de LUX-ZEPLIN (arXiv:2412.04854), que a su vez adopta el flujo PB
de Du, Fang & Liu (arXiv:2211.11469). Ese es el test integral de toda la
cadena (kernel -> seccion eficaz -> multiplicidad -> flujo).

Calcula Phi_chi^s(mchi) integrado en energia, a eps = 0.01 (como en la
Fig. 1 de LZ), en el rango de masa 10 MeV - 1 GeV, y verifica:

  (1) Normalizacion ~ 1e-9 cm^-2 s^-1 sr^-1 a baja masa.
  (2) El "codo" en mchi = m_rho/2 ~ 0.385 GeV: para mchi > m_rho/2 el umbral
      4 mchi^2 > m_rho^2 saca la resonancia rho/omega de la integral en k^2,
      y el flujo cae abruptamente (la "sudden change near mchi = m_rho/2"
      que mencionan tanto [Du] como [LZ]).
  (3) La caida abrupta a masa alta.

Eje y (como en LZ): Total Flux [s^-1 cm^-2 sr^-1].

INCERTIDUMBRES a tener presente al comparar la NORMALIZACION con LZ:
  * Lambda (form factor off-shell): [Du] dice que la produccion PB fluctua
    ~1 orden de magnitud para Lambda in [1,2] GeV. Usamos el central 1.5 GeV.
  * LZ cita 46% de incertidumbre propia en el flujo PB.
  * Factor 2 (chi vs chi+chibar): ver nota en pb_surface_flux.py.
  Por eso un acuerdo a factor ~2 en normalizacion ya es satisfactorio; la
  FORMA (meseta + codo en m_rho/2 + caida) es el check mas robusto.
"""
#%%
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
#%%
from pb_surface_flux import surface_flux, M_P

EPS = 0.01
M_RHO = 0.77

# Grillas: convergencia ~pocos % (suficiente frente al 46% de LZ y al ~1
# orden de magnitud de Lambda). Ver tests de convergencia en el .tex.
GRID = dict(Ep_max=1.0e4, n_Ep=40, n_k2=140, n_E0=64, n_cos=96)


def compute_curve(masses):
    r"""Flujo PB de superficie para un array de masas [GeV]."""
    flux = np.zeros_like(masses)
    for i, m in enumerate(masses):
        flux[i] = surface_flux(m, eps=EPS, **GRID)
        print(f"  mchi={m*1e3:7.1f} MeV   Phi_PB={flux[i]:.3e} cm^-2 s^-1 sr^-1")
    return flux

#%%
LZ = pd.read_csv('/home/lurishi/Escritorio/Doctorado/Mcp_gráficos_papers/Flujo_PB.csv',sep=';')
display(LZ.head(3))
#%%
if __name__ == "__main__":
    masses = np.logspace(np.log10(0.010), np.log10(1.0), 100)   # GeV
    print("Calculando flujo PB de superficie (eps=0.01)...")
    flux = compute_curve(masses)
    print(f"\nm_rho/2 = {M_RHO/2:.3f} GeV  (codo esperado)")

    # Figura estilo LZ Fig. 1 (panel superior), curva PB.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(LZ[LZ.columns[0]], LZ[LZ.columns[1]], "k--", lw=2, label="PB (LZ, arXiv:2412.04854)")
    ax.plot(masses * 1e3, flux, "r-", lw=2, label="PB (este trabajo)")
    ax.axvline(M_RHO / 2 * 1e3, color="gray", ls=":", lw=1,
               label=r"$m_\rho/2 \approx 385$ MeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_\chi$ [MeV/$c^2$]")
    ax.set_ylabel(r"Total Flux $\Phi_\chi$ [s$^{-1}$ cm$^{-2}$ sr$^{-1}$]")
    ax.set_title(r"Flujo PB de superficie, $\epsilon=0.01$  (comparar con LZ Fig. 1 sup.)")
    ax.set_xlim(10, 1000)
    ax.set_ylim(1e-15, 1e-7)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig("LZ_PB_flux_reproduction.png", dpi=150)
    print("\nGuardado: LZ_PB_flux_reproduction.png")

    # Tabla de referencia (para digitalizar/comparar con LZ).
    np.savetxt("PB_surface_flux_eps0.01.csv",
               np.column_stack([masses, flux]),
               header="mchi_GeV, Phi_PB_cm-2_s-1_sr-1 (eps=0.01)",
               delimiter=",")
    print("Guardado: PB_surface_flux_eps0.01.csv")




# %%
