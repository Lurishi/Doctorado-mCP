# -*- coding: utf-8 -*-
r"""
pb_surface_flux.py
==================

Flujo de superficie de mCP por proton bremsstrahlung (PB), construido
sobre el splitting kernel de pb_splitting_kernel.py.

REFERENCIAS
-----------
    [Du]  M. Du, R. Fang, Z. Liu, JHEP 08 (2024) 174, arXiv:2211.11469.
    [LZ]  LUX-ZEPLIN Collab., arXiv:2412.04854.
    [Gai] T.K. Gaisser, R. Engel, E. Resconi, "Cosmic Rays and Particle
          Physics", 2nd ed. (2016)  ([Du] ref. [110], para la cascada).

CADENA DE CALCULO
-----------------
    kernel_CM (Du 3.6-3.7)
      -> seccion eficaz  d sigma_PB/dEk        (Du 3.8)
      -> multiplicidad   N_chi por interaccion (Du 3.1, integrada en Echi)
      -> flujo de superficie  Phi_chi^s(mchi)  (Du 2.1-2.3)

COLAPSO DE LA CASCADA (clave)
-----------------------------
El flujo de superficie es (Du, Ec. 2.1):

    d^2 Phi_chi^s/(dEchi dOmega)
        = INT dh dEp  [d^2Phi_p(h)/(dEp dOmega)]  n_T(h) sigma_pT  sum_i dN^i_chi/dEchi

con el flujo de protones a altura h dado por la cascada (Du, Ec. 2.3):

    d/dh [ Phi_p(h) ] = sigma_pT n_T(h) Phi_p(h)
        =>  Phi_p(h) = Phi_p(h_max) exp( -sigma_pT INT_h^{h_max} n_T dh' ).

La integral de altura de la produccion colapsa. Cambiando variable a la
profundidad de columna u = sigma_pT INT_h^{h_max} n_T dh':

    INT dh  sigma_pT n_T(h) Phi_p(h)  =  Phi_p(h_max) (1 - e^{-tau}),

con tau = sigma_pT * (columna total) >> 1 (la atmosfera son ~11 longitudes
de interaccion para protones), de modo que 1 - e^{-tau} ~= 1. Es decir:
**cada proton interactua una vez (limite beam-dump) y el perfil atmosferico
n_T(h) se cae**; no hace falta NRLMSISE-00. Ademas sigma_pT(s2')/sigma_pT ~ 1
(seccion casi constante, 253 mb, Du bajo Ec. 2.1). Por lo tanto:

    d Phi_chi^s/(dEchi dOmega) = INT dEp  Phi_p(Ep)  dN_chi/dEchi.

Como la distribucion en Echi (Du 3.1) es plana entre E_- y E_+ y
INT dEchi [Theta Theta/(E_+ - E_-)] = 1, el flujo INTEGRADO en energia es:

    Phi_chi^s(mchi) = INT dEp  Phi_p(Ep)  N_chi(Ep, mchi),

con la MULTIPLICIDAD de mCP por interaccion (Du 3.1 integrada en Echi):

    N_chi = (eps^2 e^2 / 6 pi^2) INT_{4mchi^2}^{k2max} dk^2/k^2
              sqrt(1 - 4mchi^2/k^2) (1 + 2mchi^2/k^2) |F_V(k^2)|^2  I(k^2,s),

    I(k^2,s) = INT dEk0 dcos0  (d^2P/dEk0 dcos0) |F_*(pp-k)|^2      (en el CM).

La 2D I se integra en variables del CM (limites simples: Ek0 in [m_k, Ek0max],
cos0 in [-1,1]); F_* se evalua mapeando cada punto (Ek0,cos0) al lab, y se usa
la INVARIANCIA de d^2P entre frames (d^2P/dEk0 dcos0 dEk0 dcos0 =
d^2P/dEk dcos dEk dcos), lo que evita arrastrar el Jacobiano explicitamente.

DISTRIBUCION EN ENERGIA (para el paso siguiente, no usada aca)
--------------------------------------------------------------
Para el flujo DIFERENCIAL dPhi/dEchi (necesario para atenuar a SNOLAB) hay
que retener la distribucion plana. Los limites (Du, texto bajo 3.1) son
    E_± = gamma (Echi^r ± beta pchi^r),
con, en el reposo del gamma*: Echi^r = m_k/2, pchi^r = (m_k/2) sqrt(1-4mchi^2/k^2),
y gamma = Ek/m_k, gamma beta = |k|_lab/m_k, de modo que
    E_± = Ek/2 ± (1/2) sqrt(Ek^2 - k^2) sqrt(1 - 4mchi^2/k^2),
    E_+ - E_- = sqrt(Ek^2 - k^2) sqrt(1 - 4mchi^2/k^2).

FACTOR 2 (chi vs chi+chibar): PENDIENTE DE FIJAR CONTRA LZ
---------------------------------------------------------
Implementamos la Ec. (3.1) TAL CUAL (prefactor eps^2 e^2/6pi^2, sin factor 2
extra). En MD, [Du] pone un "2" explicito por los dos mCP del par (Ec. A.1).
Hay un argumento de que el 6pi^2 (en vez del 12pi^2 natural de Im Pi) ya
incluye el par; esto se debe pinchar superponiendo contra LZ.
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
#   d^2Phi_p/(dEp dOmega)(h_max) = 0.74 * 1.8e4 / (m^2 s sr GeV) * (Ep/GeV)^-2.7
# Pasamos m^-2 -> cm^-2 dividiendo por 1e4:
#   0.74 * 1.8e4 / 1e4 = 1.332  [cm^-2 s^-1 sr^-1 GeV^-1]
PHI_P_NORM = 0.74 * 1.8e4 / 1.0e4     # = 1.332
PHI_P_INDEX = -2.7


def proton_flux(Ep):
    r"""Flujo de protones en el tope [cm^-2 s^-1 sr^-1 GeV^-1]  [Du, Ec. 2.2]."""
    return PHI_P_NORM * np.asarray(Ep, dtype=float) ** PHI_P_INDEX


# ---------------------------------------------------------------------------
# Integral interna I(k^2, s, Ep)  (2D en el CM)  con F_* (off-shell)
# ---------------------------------------------------------------------------
def inner_integral(s, Ep, mk, n_E0=80, n_cos=120, cos_stretch=True):
    r"""I(k^2,s) = INT dEk0 dcos0 (d^2P/dEk0 dcos0) |F_*(pp-k)|^2  [adimensional].

    Es la parte de la Ec. (3.8) de [Du] con el kernel y el form factor off-shell,
    integrada en las variables del CM (limites simples). F_* se evalua en el
    lab: p'^2 = (pp - k)^2, con
        p'^2 = mp^2 + m_k^2 - 2 Ep Ek_lab + 2 |pp| |k|_lab cos_lab,
        |pp| = sqrt(Ep^2 - mp^2).

    (El |F_V|^2, que depende solo de k^2, se saca afuera y se aplica en la
     integral en k^2 dentro de multiplicity().)

    n_E0, n_cos : puntos de las grillas en Ek0 y cos0.
    cos_stretch : si True, concentra puntos hacia cos0 -> +1 (direccion
                  colineal, donde el integrando se pica a Ep alto).
    """
    E0max = Ek0_max(s, mk)
    if E0max <= mk:
        return 0.0

    # Grilla en Ek0 in [m_k, Ek0max].
    E0 = np.linspace(mk, E0max, n_E0)

    # Grilla en cos0 in [-1, 1], opcionalmente estirada hacia +1.
    if cos_stretch:
        u = np.linspace(0.0, 1.0, n_cos)
        cos0 = -1.0 + 2.0 * u**1.5        # densidad extra cerca de cos0=+1
    else:
        cos0 = np.linspace(-1.0, 1.0, n_cos)

    E0g, cos0g = np.meshgrid(E0, cos0, indexing="ij")

    # Kernel CM (Du 3.6-3.7).
    K = kernel_CM(s, E0g, cos0g, mk)

    # Mapeo CM -> lab para el argumento de F_* (p'^2 en el lab).
    Ek_lab, cos_lab, kmag_lab, _ = cm_to_lab(E0g, cos0g, s, mk)
    ppz = np.sqrt(max(Ep**2 - M_P**2, 0.0))               # |pp| en el lab
    p2 = M_P**2 + mk**2 - 2.0 * Ep * Ek_lab + 2.0 * ppz * kmag_lab * cos_lab
    Fstar2 = F_star(p2) ** 2                              # |F_*|^2

    integrand = K * Fstar2
    # Integracion 2D por trapecios: primero en cos0 (eje 1), luego en E0 (eje 0).
    I_cos = _trapz(integrand, cos0, axis=1)
    return _trapz(I_cos, E0)


# ---------------------------------------------------------------------------
# Grilla en k^2 con densidad extra en la resonancia rho/omega
# ---------------------------------------------------------------------------
def build_k2_grid(k2min, k2max, n=160):
    r"""Grilla en k^2 con puntos extra alrededor de m_rho^2 ~ 0.593 GeV^2.

    |F_V|^2 (Du 3.9) tiene un pico agudo en k^2 ~ m_rho^2 (la omega es muy
    angosta, Gamma_omega = 0.0085), que hay que resolver bien: es lo que
    produce el "codo" del flujo en m_chi = m_rho/2. (Convergencia verificada:
    N_chi estable a <1% entre n_k2=160 y 400.)
    """
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
    r"""N_chi por interaccion proton-aire (adimensional), Ec. (3.1) integrada.

        N_chi = (eps^2 e^2 / 6 pi^2) INT_{4mchi^2}^{k2max} dk^2/k^2
                  sqrt(1-4mchi^2/k^2)(1+2mchi^2/k^2) |F_V(k^2)|^2  I(k^2,s).

    El factor sqrt(1-4mchi^2/k^2)(1+2mchi^2/k^2) es el espacio de fases /
    elemento de matriz de gamma*(k^2) -> chi chibar.

    Limites en k^2:
        * inferior: k2min = 4 mchi^2  (umbral de produccion del par).
        * superior: k2max = (sqrt(s) - 2 mp)^2, de la condicion s_k >= 4mp^2
          en Ek0 = m_k (foton en reposo en el CM) => m_k <= sqrt(s) - 2 mp.
    """
    s = s_from_Ep(Ep)
    sqs = np.sqrt(s)
    mk_max = sqs - 2.0 * M_P                       # m_k maximo cinematico
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
        # Espacio de fases de gamma* -> chi chibar.
        ps = (1.0 / k2) * np.sqrt(max(1.0 - 4.0*mchi**2/k2, 0.0)) \
             * (1.0 + 2.0*mchi**2/k2)
        FV2 = np.abs(F_V(k2))**2                   # |F_V|^2 (Du 3.9)
        I = inner_integral(s, Ep, mk, n_E0=n_E0, n_cos=n_cos)
        integ[i] = ps * FV2 * I

    # Prefactor eps^2 e^2 / 6 pi^2  (Du 3.1) e integracion en k^2.
    N = (eps**2 * E2 / (6.0 * np.pi**2)) * _trapz(integ, k2grid)
    return N


# ---------------------------------------------------------------------------
# Flujo de superficie integrado en energia  Phi_chi^s(mchi)   [Du, Ec. 2.1]
# ---------------------------------------------------------------------------
def surface_flux(mchi, eps=0.01, Ep_max=1.0e4, n_Ep=48,
                 n_k2=160, n_E0=80, n_cos=120, return_curve=False):
    r"""Phi_chi^s(mchi) [cm^-2 s^-1 sr^-1], integrado en energia.

        Phi_chi^s(mchi) = INT dEp  Phi_p(Ep)  N_chi(Ep, mchi).

    Umbral en Ep: se necesita sqrt(s) >= 2 mp + 2 mchi (para tener
    k2max = (sqrt(s)-2mp)^2 >= 4 mchi^2), i.e.
        Ep >= Ep_thr = ((2mp+2mchi)^2 - 2mp^2)/(2 mp).

    Convergencia (verificada): la integral esta dominada por protones de
    baja energia (~50% del flujo de Ep<8 GeV, ~90% de Ep<30 GeV), tal como
    enfatiza [Du]; Ep_max=1e4 GeV captura ~99.5%.

    return_curve=True devuelve tambien (Ep, N_chi(Ep)) para diagnostico.
    """
    s_thr = (2.0 * M_P + 2.0 * mchi)**2
    Ep_thr = (s_thr - 2.0 * M_P**2) / (2.0 * M_P)
    if Ep_thr >= Ep_max:
        return (0.0, None, None) if return_curve else 0.0

    # Grilla log en Ep desde el umbral hasta Ep_max.
    Ep = np.logspace(np.log10(Ep_thr * 1.0005), np.log10(Ep_max), n_Ep)
    Nvals = np.array([multiplicity(E, mchi, eps=eps,
                                   n_k2=n_k2, n_E0=n_E0, n_cos=n_cos)
                      for E in Ep])
    integrand = proton_flux(Ep) * Nvals           # cm^-2 s^-1 sr^-1 GeV^-1
    Phi = _trapz(integrand, Ep)
    if return_curve:
        return Phi, Ep, Nvals
    return Phi


if __name__ == "__main__":
    # Chequeo rapido: multiplicidad y un punto de flujo.
    for mchi in (0.05, 0.1, 0.3):
        N = multiplicity(50.0, mchi, eps=0.01)
        print(f"mchi={mchi:.3f} GeV:  N_chi(Ep=50, eps=0.01) = {N:.3e}")
    print("Phi(mchi=0.1, eps=0.01) =",
          f"{surface_flux(0.1, eps=0.01, n_Ep=32):.3e} cm^-2 s^-1 sr^-1")

# %%
