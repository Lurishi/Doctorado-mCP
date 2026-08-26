# -*- coding: utf-8 -*-
r"""
validate_fig5_du.py
===================

Test de aceptacion del splitting kernel: reproduce la Fig. 5 de
Du, Fang & Liu (arXiv:2211.11469) -> el kernel  d^2P/(dEk dcos)  en el LAB,
SIN el form factor del proton (F_* = 1), para:

    Ep = 5 y 9 GeV,   sqrt(k^2) = 0.77 GeV,   cos = 1, 0.99, 0.9.

La Fig. 5 es el test limpio del kernel porque no involucra ni F_V ni F_*
ni la integral en k^2: es directamente el kernel (Du 3.6-3.7) mas el boost.

Chequeos que produce
--------------------
  * Endpoints cinematicos Ek^max(cos=1): deben coincidir con el rango del
    eje x de cada panel (~3.5 GeV para Ep=5, ~8 GeV para Ep=9). Esto valida
    la eleccion s = 2 mp Ep + 2 mp^2 (s_from_Ep).
  * Compara tres prescripciones de Jacobiano ('full', 'kmag', 'none'):
    'full' y 'kmag' deben COINCIDIR (verifica que el Jacobiano 2D completo
    se reduce al |k|/|k0| de Du, Ec. 3.8).

OBS de lectura de la Fig. 5: la curva mas alta del panel es la FWW (roja
punteada), que segun el Apendice B de [Du] es mayor que el "new method" en
cos=1. Al comparar hay que contrastar contra las curvas SOLIDAS (New PB),
que son las que reproduce este kernel.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pb_splitting_kernel import (
    s_from_Ep, gamma_beta_i, Ek0_max, cm_to_lab, kernel_lab,
)

MK = 0.77                              # sqrt(k^2) [GeV], como en la Fig. 5
CURVES = [(1.00, "red"), (0.99, "blue"), (0.90, "green")]


def ek_lab_max(Ep, mk):
    r"""Ek maximo en el lab (cos0 = +1, Ek0 = Ek0_max)."""
    s = s_from_Ep(Ep)
    Ek, _, _, _ = cm_to_lab(Ek0_max(s, mk), 1.0, s, mk)
    return float(Ek)


def make_panel(ax, Ep, ek_max_axis, jac="full"):
    r"""Dibuja un panel (Ep fijo) con las curvas cos=1, 0.99, 0.9."""
    s = s_from_Ep(Ep)
    Ek = np.linspace(MK * 1.0001, ek_max_axis, 800)
    for cos_lab, color in CURVES:
        K = kernel_lab(s, Ek, np.full_like(Ek, cos_lab), MK, jacobian=jac)
        K = np.where(K > 0, K, np.nan)
        ax.plot(Ek, K, color=color, lw=1.6,
                label=fr"$\cos\theta={cos_lab}$ (New PB)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_k$ [GeV]")
    ax.set_ylabel(r"$d^2\mathcal{P}_{p\to\gamma^*p}/(dE_k\,d\cos\theta_k)$")
    ax.set_ylim(1e-4, 1e-1)
    ax.set_xlim(0, ek_max_axis)
    ax.set_title(fr"$E_p={Ep:.0f}$ GeV @ lab,  $\sqrt{{k^2}}=0.77$ GeV  [{jac}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


if __name__ == "__main__":
    # Endpoints cinematicos (deben calzar con los ejes de la Fig. 5).
    for Ep in (5.0, 9.0):
        s = s_from_Ep(Ep)
        gi, bi = gamma_beta_i(s)
        print(f"Ep={Ep:.0f} GeV: s={s:.3f} GeV^2, sqrt(s)={np.sqrt(s):.3f}, "
              f"gamma_i={gi:.3f}, beta_i={bi:.3f}, "
              f"Ek0_max={Ek0_max(s, MK):.3f} GeV, "
              f"Ek_lab_max(cos=1)={ek_lab_max(Ep, MK):.3f} GeV")

    # Una figura por prescripcion de Jacobiano ('full' y 'kmag' deben coincidir).
    for jac in ("full", "kmag", "none"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        make_panel(axes[0], 5.0, 3.6, jac=jac)
        make_panel(axes[1], 9.0, 8.0, jac=jac)
        plt.tight_layout()
        fname = f"fig5_du_reproduction_{jac}.png"
        plt.savefig(fname, dpi=140)
        plt.close(fig)
        print("Guardado:", fname)
