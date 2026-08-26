# -*- coding: utf-8 -*-
r"""
pb_differential_flux.py
========================

Flujo DIFERENCIAL en energia de mCP por proton bremsstrahlung (PB):

    d Phi_chi^s / dEchi   [cm^-2 s^-1 sr^-1 GeV^-1]

Extiende pb_surface_flux.py reteniendo la distribucion en Echi, que [Du]
(texto bajo la Ec. 3.1) toma PLANA entre E_- y E_+ para cada punto cinematico
(k^2, Ek0, cos0):

    dN_chi/dEchi |_{k^2,Ek0,cos0} = Theta(Echi - E_-) Theta(E_+ - Echi) / (E_+ - E_-)

con, en el reposo del gamma* (masa m_k = sqrt(k^2)):

    Echi^r = m_k/2 ,          pchi^r = (m_k/2) sqrt(1 - 4 mchi^2/k^2) ,

y boosteando al lab con gamma = Ek/m_k, gamma*beta = |k|_lab/m_k (Ek, |k|_lab
son la energia y el momento del foton off-shell EN EL LAB para ese punto
(Ek0,cos0) del CM, mapeados con cm_to_lab):

    E_pm = Ek/2 +- (1/2) sqrt(Ek^2 - k^2) sqrt(1 - 4 mchi^2/k^2)
         = Ek/2 +- (1/2) |k|_lab sqrt(1 - 4 mchi^2/k^2) .

CADENA DE CALCULO
------------------
Partiendo de la formula maestra (Du 3.1) SIN integrar en Echi:

    dN_chi/dEchi(Ep,mchi;Echi) =
        (eps^2 e^2 / 6 pi^2) INT dk^2/k^2 sqrt(1-4mchi^2/k^2)(1+2mchi^2/k^2)
                              |F_V(k^2)|^2  J(k^2,s,Echi) ,

    J(k^2,s,Echi) = INT dEk0 dcos0  (d^2P/dEk0 dcos0) |F_*(pp-k)|^2
                     * Theta(E_- <= Echi <= E_+) / (E_+ - E_-) ,

(la integral J es la version "resuelta en Echi" de I(k^2,s) de
pb_surface_flux.inner_integral; INT dEchi J(k^2,s,Echi) = I(k^2,s) porque la
ventana Theta/(E_+-E_-) integra a 1 en Echi). Y finalmente

    d Phi_chi^s/dEchi (mchi;Echi) = INT dEp  Phi_p(Ep)  dN_chi/dEchi(Ep,mchi;Echi).

CONSISTENCIA (test de aceptacion de este modulo)
-------------------------------------------------
Por construccion,

    INT dEchi  d Phi_chi^s/dEchi (mchi;Echi)  =  Phi_chi^s(mchi)

el flujo INTEGRADO de pb_surface_flux.surface_flux(). Este modulo se valida
comprobando esa igualdad (ver check_consistency() abajo y el bloque
__main__): con grillas moderadas coincide a nivel de pocos % (mismo orden
de convergencia que ya reportado para I(k^2,s) e Ip surface_flux en el .tex).

METODO NUMERICO: REPARTO CONSERVATIVO (regridding), no un Theta puntual
------------------------------------------------------------------------
La primera version de este modulo evaluaba, en una grilla FIJA de Echi, el
indicador booleano Theta(E_- <= Echi <= E_+) en cada punto (Ek0,cos0) de la
grilla CM y despues integraba con trapz. Eso funciona MAL: el borde de la
ventana [E_-,E_+] es una funcion escalon en (Ek0,cos0), y el trapz sobre una
grilla regular converge solo linealmente en el numero de puntos frente a un
escalon (a diferencia de la convergencia rapida que trapz logra en funciones
suaves). Se verifico numericamente: con n_Echi hasta 8000 el resultado
seguia ~7-8% por debajo del flujo integrado de referencia (pb_surface_flux),
y solo mejoraba refinando ademas la grilla (Ek0,cos0) -- pero eso es lento.

La solucion adoptada aca es DEPOSITO CONSERVATIVO (analogo a "conservative
regridding" en grillas de clima/oceanografia): cada celda de la grilla 2D
(Ek0,cos0) tiene un peso trapezoidal exacto w_cell = (K * |F_*|^2) * dE0_cell
* dcos0_cell (estos w_cell SUMAN EXACTAMENTE la misma I(k^2,s) que calcula
pb_surface_flux.inner_integral con trapz estandar). Esa celda contribuye una
densidad plana de altura w_cell/(E_+-E_-) sobre el intervalo [E_-,E_+] en
Echi. En vez de evaluarla en puntos sueltos de Echi, se reparte su integral
EXACTAMENTE entre los bines de salida en Echi calculando la longitud de
solape analitica  overlap(celda,bin) = max(0, min(edge_hi,E_+) -
max(edge_lo,E_-))  y sumando  w_cell/(E_+-E_-) * overlap  al bin. Esto:

    (a) reproduce por construccion  SUM_bins (integral del bin) = SUM_cells
        w_cell = I(k^2,s)  (mismo valor que el metodo no diferencial, hasta
        el error de la grilla (Ek0,cos0) que YA esta documentado y
        controlado en pb_surface_flux.py);
    (b) no tiene el sesgo de discretizacion en Echi descrito arriba, porque
        el solape se calcula analiticamente y no por muestreo puntual.

Con esto, unos ~40-80 bines en Echi ya alcanzan el nivel de precision de la
grilla (Ek0,cos0) subyacente (unos pocos %), sin necesitar miles de puntos.
"""
#%%
import numpy as np
import pandas as pd
LZ = pd.read_csv('/home/lurishi/Escritorio/Doctorado/PB_from_DU/PB_cos_1.csv',sep=';')
#%%
from pb_splitting_kernel import (
    ALPHA, M_P, F_V, F_star, kernel_CM, Ek0_max, s_from_Ep, cm_to_lab,
)
from pb_surface_flux import E2, proton_flux, build_k2_grid

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid        # numpy >= 2.0
else:
    _trapz = np.trapz            # numpy < 2.0


def _trapz_weights(x):
    r"""Pesos de cuadratura trapezoidal para una grilla 1D (posiblemente no
    uniforme): w_i tal que  SUM_i f_i w_i ~= INT f dx  (equivalente a trapz).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = np.zeros(n)
    if n == 1:
        return np.array([1.0])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if n > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _echi_bin_edges(Echi_centers):
    r"""Construye bordes de bin a partir de centros (para el reparto), usando
    el punto medio entre centros consecutivos y extrapolando en los extremos.
    """
    c = np.asarray(Echi_centers, dtype=float)
    mid = 0.5 * (c[1:] + c[:-1])
    left = c[0] - (mid[0] - c[0])
    right = c[-1] + (c[-1] - mid[-1])
    edges = np.concatenate([[left], mid, [right]])
    return edges


# ---------------------------------------------------------------------------
# Integral interna J(k^2, s, Echi)  (2D en el CM, reparto conservativo)
# ---------------------------------------------------------------------------
def inner_integral_diff(s, Ep, mk, mchi, Echi_edges, n_E0=60, n_cos=80,
                         cos_stretch=True):
    r"""Integral de J(k^2,s,Echi) EN CADA BIN de Echi (bordes Echi_edges),
    repartida conservativamente desde la grilla (Ek0,cos0).

    Devuelve un array de longitud len(Echi_edges)-1 con el valor INTEGRADO
    de J en cada bin (dividir por el ancho del bin para obtener la densidad).
    """
    Echi_edges = np.asarray(Echi_edges, dtype=float)
    n_bins = len(Echi_edges) - 1

    E0max = Ek0_max(s, mk)
    if E0max <= mk:
        return np.zeros(n_bins)

    ratio = 1.0 - 4.0 * mchi**2 / mk**2
    if ratio <= 0.0:
        return np.zeros(n_bins)
    sqrt_ratio = np.sqrt(ratio)

    # Grilla en Ek0 in [m_k, Ek0max] y cos0 in [-1,1] (igual que inner_integral).
    E0 = np.linspace(mk, E0max, n_E0)
    if cos_stretch:
        u = np.linspace(0.0, 1.0, n_cos)
        cos0 = -1.0 + 2.0 * u**1.5
    else:
        cos0 = np.linspace(-1.0, 1.0, n_cos)

    E0g, cos0g = np.meshgrid(E0, cos0, indexing="ij")           # (n_E0,n_cos)

    K = kernel_CM(s, E0g, cos0g, mk)

    Ek_lab, cos_lab, kmag_lab, _ = cm_to_lab(E0g, cos0g, s, mk)

    ppz = np.sqrt(max(Ep**2 - M_P**2, 0.0))
    p2 = M_P**2 + mk**2 - 2.0*Ep*Ek_lab + 2.0*ppz*kmag_lab*cos_lab
    Fstar2 = F_star(p2) ** 2

    Ewidth = kmag_lab * sqrt_ratio          # E_+ - E_-           (n_E0,n_cos)
    Eplus  = 0.5 * (Ek_lab + Ewidth)
    Eminus = 0.5 * (Ek_lab - Ewidth)

    # Peso trapezoidal EXACTO de cada celda 2D (reproduce el trapz estandar).
    wE0 = _trapz_weights(E0)
    wcos = _trapz_weights(cos0)
    cell_area = np.outer(wE0, wcos)                 # (n_E0,n_cos)
    w_cell = K * Fstar2 * cell_area                  # peso integrado por celda

    valid = Ewidth > 0
    w_flat = w_cell[valid]
    Emin_flat = Eminus[valid]
    Emax_flat = Eplus[valid]
    Ewid_flat = Ewidth[valid]
    if w_flat.size == 0:
        return np.zeros(n_bins)

    # Reparto conservativo: solape de cada celda con cada bin de Echi.
    #   overlap(cell,bin) = max(0, min(edge_hi,E+) - max(edge_lo,E-))
    edge_lo = Echi_edges[:-1].reshape(1, -1)          # (1,n_bins)
    edge_hi = Echi_edges[1:].reshape(1, -1)           # (1,n_bins)
    lo = np.maximum(Emin_flat[:, None], edge_lo)      # (n_cells,n_bins)
    hi = np.minimum(Emax_flat[:, None], edge_hi)
    overlap = np.clip(hi - lo, 0.0, None)

    contrib = (w_flat / Ewid_flat)[:, None] * overlap  # integral por bin, por celda
    J_bins = contrib.sum(axis=0)                       # (n_bins,)
    return J_bins


# ---------------------------------------------------------------------------
# dN_chi/dEchi(Ep,mchi;Echi)   [Du, Ec. 3.1 sin integrar en Echi]
# ---------------------------------------------------------------------------
def differential_multiplicity(Ep, mchi, Echi, eps=1.0, n_k2=100,
                               n_E0=60, n_cos=80):
    r"""dN_chi/dEchi por interaccion proton-aire [1/GeV], en los puntos Echi
    (array de CENTROS; internamente se construyen bordes de bin para el
    reparto conservativo y se devuelve la densidad en cada centro).
    """
    Echi = np.atleast_1d(np.asarray(Echi, dtype=float))
    edges = _echi_bin_edges(Echi)
    widths = np.diff(edges)

    s = s_from_Ep(Ep)
    sqs = np.sqrt(s)
    mk_max = sqs - 2.0 * M_P
    if mk_max <= 2.0 * mchi:
        return np.zeros_like(Echi)
    k2min = 4.0 * mchi**2
    k2max = mk_max**2
    if k2min >= k2max:
        return np.zeros_like(Echi)

    k2grid = build_k2_grid(k2min, k2max, n_k2)
    integ = np.zeros((len(k2grid), len(Echi)))
    for i, k2 in enumerate(k2grid):
        mk = np.sqrt(k2)
        ps = (1.0/k2) * np.sqrt(max(1.0 - 4.0*mchi**2/k2, 0.0)) \
             * (1.0 + 2.0*mchi**2/k2)
        FV2 = np.abs(F_V(k2))**2
        Jbins = inner_integral_diff(s, Ep, mk, mchi, edges,
                                     n_E0=n_E0, n_cos=n_cos)
        integ[i, :] = ps * FV2 * Jbins / widths       # densidad en Echi

    N = (eps**2 * E2 / (6.0 * np.pi**2)) * _trapz(integ, k2grid, axis=0)
    return N


# ---------------------------------------------------------------------------
# dPhi_chi^s/dEchi(mchi;Echi)   [Du, Ec. 2.1, colapsada, sin integrar Echi]
# ---------------------------------------------------------------------------
def differential_surface_flux(mchi, Echi, eps=0.01, Ep_max=1.0e4, n_Ep=30,
                                n_k2=100, n_E0=60, n_cos=80):
    r"""dPhi_chi^s/dEchi(mchi) [cm^-2 s^-1 sr^-1 GeV^-1], en los puntos Echi
    (array de centros de bin)."""
    Echi = np.atleast_1d(np.asarray(Echi, dtype=float))
    s_thr = (2.0 * M_P + 2.0 * mchi)**2
    Ep_thr = (s_thr - 2.0 * M_P**2) / (2.0 * M_P)
    if Ep_thr >= Ep_max:
        return np.zeros_like(Echi)

    Ep_grid = np.logspace(np.log10(Ep_thr * 1.0005), np.log10(Ep_max), n_Ep)
    dPhi = np.zeros((n_Ep, len(Echi)))
    for j, Ep in enumerate(Ep_grid):
        dN = differential_multiplicity(Ep, mchi, Echi, eps=eps,
                                        n_k2=n_k2, n_E0=n_E0, n_cos=n_cos)
        dPhi[j, :] = proton_flux(Ep) * dN

    result = _trapz(dPhi, Ep_grid, axis=0)
    return result


# ---------------------------------------------------------------------------
# Test de consistencia: INT dEchi dPhi/dEchi  ==  surface_flux (integrado)
# ---------------------------------------------------------------------------
def check_consistency(mchi, eps=0.01, n_Echi=120, verbose=True, **grid_kwargs):
    r"""Compara INT dEchi dPhi/dEchi(mchi;Echi) contra pb_surface_flux.surface_flux.

    Devuelve (Phi_integrated_from_diff, Phi_direct, ratio).
    """
    from pb_surface_flux import surface_flux

    # Rango de Echi: de mchi hasta un maximo generoso (Ek_lab puede llegar a
    # energias de Ep; unos pocos GeV alcanza para capturar >99% del soporte
    # a mchi sub-GeV, ya que el flujo de protones cae como Ep^-2.7).
    Echi_max = 1e2#max(50.0, 30.0 * mchi + 5.0)
    Echi = np.linspace(mchi * 1.0001, Echi_max, n_Echi)

    dPhi = differential_surface_flux(mchi, Echi, eps=eps, **grid_kwargs)
    Phi_from_diff = _trapz(dPhi, Echi)

    Phi_direct = surface_flux(mchi, eps=eps,
                               Ep_max=grid_kwargs.get("Ep_max", 1.0e4),
                               n_Ep=grid_kwargs.get("n_Ep", 30),
                               n_k2=grid_kwargs.get("n_k2", 100),
                               n_E0=grid_kwargs.get("n_E0", 60),
                               n_cos=grid_kwargs.get("n_cos", 80))

    ratio = Phi_from_diff / Phi_direct if Phi_direct > 0 else np.nan
    if verbose:
        print(f"mchi={mchi*1e3:.1f} MeV:  "
              f"INT dEchi dPhi/dEchi = {Phi_from_diff:.4e}   "
              f"surface_flux directo = {Phi_direct:.4e}   "
              f"ratio = {ratio:.4f}")
    return Phi_from_diff, Phi_direct, ratio

#%%
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EPS = 0.01
    GRID = dict(Ep_max=1.0e4, n_Ep=24, n_k2=80, n_E0=40, n_cos=60)

    print("Test de consistencia (INT dEchi dPhi/dEchi vs surface_flux):")
    for mchi in (0.03, 0.1, 0.3):
        check_consistency(mchi, eps=EPS, n_Echi=100, **GRID)

    print("\nCalculando espectros dPhi/dEchi...")
    fig, ax = plt.subplots(figsize=(7, 6))
    for mchi, color in [(0.03, "C0"), (0.1, "C1"), (0.3, "C2")]:
        Echi_max = 1e2#max(30.0, 25.0 * mchi + 5.0)
        Echi = np.linspace(mchi * 1.0001, Echi_max, 150)
        dPhi = differential_surface_flux(mchi, Echi, eps=EPS, **GRID)
        ax.plot(Echi, dPhi, color=color, lw=1.8,
                label=fr"$m_\chi={mchi*1e3:.0f}$ MeV")
    ax.plot(np.sort(LZ[LZ.columns[0]].values), np.sort(LZ[LZ.columns[1]].values)[::-1], "k--", lw=2, label="PB (LZ, arXiv:2412.04854)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_\chi$ [GeV]")
    ax.set_ylabel(r"$d\Phi_\chi^s/dE_\chi$  [cm$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$]")
    ax.set_title(fr"Flujo diferencial PB de superficie, $\epsilon={EPS}$")

    ax.legend()
    ax.grid(True,linestyle = '--', which="both")
    plt.tight_layout()
    plt.savefig("PB_differential_flux.png", dpi=150)
    print("Guardado: PB_differential_flux.png")


# %%
