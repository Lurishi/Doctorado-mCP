# -*- coding: utf-8 -*-
r"""
pb_splitting_kernel.py
======================

Generador del *splitting kernel*  p -> gamma* p  para la produccion de
particulas milicargadas (mCP) por proton bremsstrahlung (PB) en la
atmosfera.

REFERENCIA PRINCIPAL
--------------------
    [Du]  M. Du, R. Fang, Z. Liu,
          "Millicharged particles from proton bremsstrahlung in the
           atmosphere", JHEP 08 (2024) 174, arXiv:2211.11469.

    [LZ]  J. Aalbers et al. (LUX-ZEPLIN),
          "First constraint on atmospheric millicharged particles with
           the LUX-ZEPLIN experiment", arXiv:2412.04854.
          (LZ adopta el flujo PB justamente de [Du].)

Este modulo implementa la parte de FISICA DE PRODUCCION del "new method"
de [Du], Sec. 3, hasta el nivel del kernel:

    * kernel_CM   ->  d^2 P / (dEk0 dcos0)    [Du, Ec. (3.6) + (3.7)]  (frame CM)
    * F_V         ->  form factor vectorial time-like   [Du, Ec. (3.9)]
    * F_star      ->  form factor off-shell del proton  [Du, Ec. (3.10)]
    * cm_to_lab / lab_to_cm  ->  boost del foton off-shell entre CM y lab
    * kernel_lab  ->  d^2 P / (dEk dcos)   [frame lab; para validar Fig. 5 de Du]

CONVENCIONES
------------
    * Unidades naturales de HEP, energias/momentos/masas en GeV.
    * k^2 es la virtualidad (masa invariante^2) del foton off-shell gamma*.
      m_k := sqrt(k^2)  es su "masa".
    * Las cantidades con superindice 0 estan en el frame CM del sistema
      p-pbar; sin superindice, en el lab (blanco en reposo).
    * El frame CM se mueve respecto al lab con (beta_i, gamma_i) a lo largo
      del eje del proton incidente. Por simetria del sistema p-pbar,
          E_p^0 = sqrt(s)/2 ,
          beta_i = sqrt(1 - 4 mp^2 / s) ,      [velocidad de c/proton en el CM]
          gamma_i = sqrt(s)/(2 mp) = 1/sqrt(1-beta_i^2).
    * s (Mandelstam del sistema p-pbar) se obtiene de la energia del proton
      en el lab por  s = 2 mp Ep + 2 mp^2  (blanco fijo). Esta eleccion se
      VALIDO comparando los endpoints cinematicos con la Fig. 5 de [Du]
      (ver validate_fig5_du.py): da Ek^max(cos=1)=3.60 GeV para Ep=5 GeV
      (eje ~3.5) y 7.61 GeV para Ep=9 GeV (eje ~8).

NOTA sobre el Jacobiano CM->lab
-------------------------------
[Du] escribe el Jacobiano de la Ec. (3.8) como |k|/|k0|. Verificamos
numericamente (kernel_lab con jacobian='full' vs 'kmag') que el Jacobiano
2D completo |d(Ek0,cos0)/d(Ek,cos)| coincide EXACTAMENTE con |k|/|k0| en
todo el dominio fisico; ese fue un cross-check clave del boost.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constantes fisicas
# ---------------------------------------------------------------------------
ALPHA = 1.0 / 137.035999            # constante de estructura fina  (alpha = e^2/4pi)
M_P   = 0.9382720813                # masa del proton [GeV]

# Corte del form factor off-shell del proton, F_* (Du, Ec. 3.10).
# [Du] considera 1 GeV < Lambda < 2 GeV y advierte que la produccion PB
# fluctua ~1 orden de magnitud en ese rango; adoptan el valor central:
LAMBDA_OFFSHELL = 1.5               # [GeV]

# Mesones vectoriales que entran en el form factor time-like F_V (Du, Ec. 3.9):
#   rho, rho', rho'', omega, omega', omega''.
# Cada entrada: (m_V [GeV], Gamma_V [GeV], f_V).
# Valores tomados literalmente del texto bajo la Ec. (3.9) de [Du]:
#   m_rho = m_omega = 0.77,  m_rho' = m_omega' = 1.25,  m_rho'' = m_omega'' = 1.45
#   Gamma_rho = 0.150, Gamma_omega = 0.0085,
#   Gamma_rho' = Gamma_omega' = 0.3,  Gamma_rho'' = Gamma_omega'' = 0.5
#   f_rho = 0.616, f_rho' = 0.223, f_rho'' = -0.339,
#   f_omega = 1.011, f_omega' = -0.881, f_omega'' = 0.369
# Chequeo de normalizacion VMD:  sum_V f_V = 0.999  =>  |F_V(0)|^2 ~= 1.
VECTOR_MESONS = [
    (0.77,  0.150,   0.616),   # rho
    (1.25,  0.30,    0.223),   # rho'
    (1.45,  0.50,   -0.339),   # rho''
    (0.77,  0.0085,  1.011),   # omega
    (1.25,  0.30,   -0.881),   # omega'
    (1.45,  0.50,    0.369),   # omega''
]


# ---------------------------------------------------------------------------
# Form factors
# ---------------------------------------------------------------------------
def F_V(k2):
    r"""Form factor vectorial time-like  [Du, Ec. (3.9)]:

        F_V(k) = sum_V  f_V m_V^2 / (m_V^2 - k^2 - i m_V Gamma_V).

    Modela el acoplamiento p-gamma*-p en la region time-like 0 < k^2 < 4mp^2
    (dominancia de mesones vectoriales, VMD). Genera el pico resonante en
    k^2 ~ m_rho^2 responsable del "codo" del flujo en m_chi = m_rho/2.

    Parametros
    ----------
    k2 : k^2 [GeV^2], escalar o array.

    Devuelve el valor COMPLEJO. Usar np.abs(F_V(k2))**2 para |F_V|^2.
    """
    k2 = np.asarray(k2, dtype=float)
    result = np.zeros(np.shape(k2), dtype=complex)
    for mV, GV, fV in VECTOR_MESONS:
        result = result + fV * mV**2 / (mV**2 - k2 - 1j * mV * GV)
    return result


def F_star(p_prime_sq):
    r"""Form factor off-shell del proton intermedio  [Du, Ec. (3.10)]:

        F_*(p') = Lambda^4 / ( Lambda^4 + (p'^2 - mp^2)^2 ),

    con p'^2 = (p_p - k)^2 el cuadrimomento^2 del proton intermedio.
    Suprime las configuraciones muy off-shell (|p'^2 - mp^2| grande).

    Parametros
    ----------
    p_prime_sq : p'^2 [GeV^2], escalar o array.

    Devuelve F_* (real, positivo, en (0,1]). En la Ec. (3.8) aparece como
    |F_*|^2, asi que al usarlo hay que elevarlo al cuadrado.
    """
    p_prime_sq = np.asarray(p_prime_sq, dtype=float)
    L4 = LAMBDA_OFFSHELL**4
    return L4 / (L4 + (p_prime_sq - M_P**2)**2)


# ---------------------------------------------------------------------------
# Kernel en el frame CM  [Du, Ec. (3.6) + (3.7)]
# ---------------------------------------------------------------------------
def kernel_CM(s, Ek0, cos0, mk):
    r"""Splitting kernel  p -> gamma* p  en el CM:  d^2P/(dEk0 dcos0)  [1/GeV].

    Implementa [Du, Ec. (3.6)]:

        d^2P/(dEk0 dcos0) = (2 alpha)/(pi Ek0)
                            * (2 beta_f beta_k)/((3 - beta_f^2) beta_i)
                            * N / ( s [x^2 - (1-y)^2]^2 ),

    con N dado por [Du, Ec. (3.7)]:

        N = (1-y)^2 [ beta_i^2 (2mp^2 + s_k + m_k^2) + Ek0^2 beta_k^2 ]
            - [ 2mp^2 + m_k^2 (1 + s_k/s) + s_k ] x^2
            - Ek0^2 x^4,

    y las definiciones (texto bajo Ec. 3.7):
        s_k    = s + m_k^2 - 2 Ek0 sqrt(s)          [= (p3+p4)^2, masa^2 del pp final]
        beta_i = sqrt(1 - 4 mp^2 / s)               [velocidad del proton inicial, CM]
        beta_f = sqrt(1 - 4 mp^2 / s_k)             [velocidad del proton final, CM]
        beta_k = sqrt(1 - m_k^2 / Ek0^2)            [velocidad del foton off-shell]
        x      = beta_i beta_k cos0
        y      = m_k^2 / (Ek0 sqrt(s))

    OBS: la Ec. (3.4) tiene un 1/sigma_2->2(s_k) que ya se CANCELO al
    obtener la forma cerrada (3.6); por eso aca no aparece.

    Umbral cinematico: s_k >= 4 mp^2 (energia suficiente para el par pp final).
    En s_k = 4 mp^2 => beta_f = 0 => Ek0 alcanza Ek0_max (ver Ek0_max()).

    Chequeo dimensional: [d^2P/dEk0 dcos0] = 1/energia. En efecto
    N ~ energia^2, s ~ energia^2, corchetes adimensionales => N/(s[...]^2)
    adimensional; (2 alpha / pi Ek0) ~ 1/energia. OK.

    Puntos NO fisicos (s_k < 4mp^2, Ek0 < m_k, etc.) devuelven 0.
    """
    Ek0  = np.asarray(Ek0, dtype=float)
    cos0 = np.asarray(cos0, dtype=float)
    mk2  = mk**2
    sqs  = np.sqrt(s)

    # beta_i: velocidad de cada proton inicial en el CM.
    beta_i2 = 1.0 - 4.0 * M_P**2 / s
    beta_i  = np.sqrt(np.clip(beta_i2, 0.0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        # s_k = (p3+p4)^2 : masa invariante^2 del sistema pp final.
        sk = s + mk2 - 2.0 * Ek0 * sqs
        beta_f2 = 1.0 - 4.0 * M_P**2 / sk
        beta_f  = np.sqrt(np.clip(beta_f2, 0.0, None))

        # beta_k: velocidad del foton off-shell en el CM (m_k = sqrt(k^2)).
        beta_k2 = 1.0 - mk2 / Ek0**2
        beta_k  = np.sqrt(np.clip(beta_k2, 0.0, None))

        # Variables cinematicas adimensionales.
        x = beta_i * beta_k * cos0
        y = mk2 / (Ek0 * sqs)

        # N  [Du, Ec. (3.7)]
        N = ((1.0 - y)**2 * (beta_i**2 * (2.0*M_P**2 + sk + mk2)
                             + Ek0**2 * beta_k**2)
             - (2.0*M_P**2 + mk2 * (1.0 + sk/s) + sk) * x**2
             - Ek0**2 * x**4)

        # Denominador y prefactor de la Ec. (3.6).
        # OBS: el polo x^2 = (1-y)^2 (singularidad colineal) NO se alcanza en
        # el dominio fisico (verificado: |x| < (1-y) siempre), asi que el
        # denominador es positivo y acotado lejos de cero.
        denom = s * (x**2 - (1.0 - y)**2)**2
        pref  = ((2.0 * ALPHA / (np.pi * Ek0))
                 * (2.0 * beta_f * beta_k) / ((3.0 - beta_f**2) * beta_i))

        kernel = pref * N / denom

    # Mascara de region fisica.
    physical = (sk >= 4.0 * M_P**2) & (Ek0 >= mk) & np.isfinite(kernel)
    kernel = np.where(physical, kernel, 0.0)
    return kernel


def Ek0_max(s, mk):
    r"""Energia maxima del foton off-shell en el CM [GeV].

    Se obtiene de la condicion s_k = 4 mp^2 (par pp final en reposo). De
    s_k = s + m_k^2 - 2 Ek0 sqrt(s) = 4 mp^2 despejando Ek0 (coincide con
    E_k^{0,max} = (4(Ep0)^2 - 4mp^2 + k^2)/(4 Ep0) de [Du], con Ep0=sqrt(s)/2):

        Ek0_max = (s - 4 mp^2 + m_k^2) / (2 sqrt(s)).
    """
    return (s - 4.0 * M_P**2 + mk**2) / (2.0 * np.sqrt(s))


# ---------------------------------------------------------------------------
# Boost CM <-> lab del foton off-shell
# ---------------------------------------------------------------------------
def s_from_Ep(Ep):
    r"""Mandelstam s del sistema p-pbar desde la energia del proton en el lab.

        s = 2 mp Ep + 2 mp^2       (proton incidente sobre nucleon en reposo).

    VALIDADO por los endpoints cinematicos de la Fig. 5 de [Du]
    (ver validate_fig5_du.py).
    """
    return 2.0 * M_P * Ep + 2.0 * M_P**2


def gamma_beta_i(s):
    r"""(gamma_i, beta_i) del boost CM->lab a lo largo del proton incidente.

        beta_i  = sqrt(1 - 4 mp^2 / s)
        gamma_i = sqrt(s) / (2 mp)  = 1/sqrt(1 - beta_i^2).
    """
    beta_i = np.sqrt(1.0 - 4.0 * M_P**2 / s)
    gamma_i = np.sqrt(s) / (2.0 * M_P)
    return gamma_i, beta_i


def cm_to_lab(Ek0, cos0, s, mk):
    r"""CM (Ek0, cos0) -> lab. Devuelve (Ek, cos_lab, |k|_lab, |k|_cm).

    Boost de Lorentz del cuadrimomento del foton off-shell a lo largo del
    eje del proton (z), con (gamma_i, beta_i):

        E_k   = gamma_i (Ek0 + beta_i kz0)
        k_z   = gamma_i (kz0 + beta_i Ek0)
        k_perp= k_perp^0   (invariante transversal)

    con kz0 = |k|_cm cos0,  |k|_cm = sqrt(Ek0^2 - m_k^2).
    """
    gamma_i, beta_i = gamma_beta_i(s)
    k0 = np.sqrt(np.clip(Ek0**2 - mk**2, 0.0, None))     # |k| en el CM
    kz0 = k0 * cos0
    kperp = k0 * np.sqrt(np.clip(1.0 - cos0**2, 0.0, None))
    Ek = gamma_i * (Ek0 + beta_i * kz0)
    kz = gamma_i * (kz0 + beta_i * Ek0)
    kmag = np.sqrt(kz**2 + kperp**2)                     # |k| en el lab
    cos_lab = np.where(kmag > 0, kz / kmag, 1.0)
    return Ek, cos_lab, kmag, k0


def lab_to_cm(Ek, cos_lab, s, mk):
    r"""lab (Ek, cos_lab) -> CM. Devuelve (Ek0, cos0, |k|_lab, |k|_cm).

    Boost inverso (beta_i -> -beta_i):

        Ek0 = gamma_i (Ek - beta_i kz)
        kz0 = gamma_i (kz - beta_i Ek)
        k_perp^0 = k_perp.
    """
    gamma_i, beta_i = gamma_beta_i(s)
    kmag = np.sqrt(np.clip(Ek**2 - mk**2, 0.0, None))    # |k| en el lab
    kz = kmag * cos_lab
    kperp = kmag * np.sqrt(np.clip(1.0 - cos_lab**2, 0.0, None))
    Ek0 = gamma_i * (Ek - beta_i * kz)
    kz0 = gamma_i * (kz - beta_i * Ek)
    k0 = np.sqrt(kz0**2 + kperp**2)                      # |k| en el CM
    cos0 = np.where(k0 > 0, kz0 / k0, 1.0)
    return Ek0, cos0, kmag, k0


# ---------------------------------------------------------------------------
# Kernel en el lab  (solo para validar contra la Fig. 5 de [Du])
# ---------------------------------------------------------------------------
def kernel_lab(s, Ek, cos_lab, mk, jacobian="full", eps=1e-6):
    r"""Splitting kernel en el lab:  d^2P/(dEk dcos_lab)  [1/GeV].

    Como P (probabilidad) es un escalar, d^2P es invariante de frame, de modo
    que la densidad diferencial doble transforma con el Jacobiano 2D:

        d^2P/(dEk dcos)  =  |d(Ek0,cos0)/d(Ek,cos)| * d^2P/(dEk0 dcos0).

    jacobian
    --------
        "full" : Jacobiano 2D completo (numerico, diferencias finitas). Es
                 lo correcto para la densidad diferencial doble en el lab.
        "kmag" : factor |k|_lab/|k|_cm que aparece en [Du, Ec. (3.8)].
                 (Verificamos que 'full' == 'kmag' numericamente.)
        "none" : sin Jacobiano (kernel CM evaluado en el punto mapeado).
    """
    Ek = np.atleast_1d(np.asarray(Ek, dtype=float))
    cos_lab = np.atleast_1d(np.asarray(cos_lab, dtype=float))
    Ek, cos_lab = np.broadcast_arrays(Ek, cos_lab)

    Ek0, cos0, kmag, k0 = lab_to_cm(Ek, cos_lab, s, mk)
    kern = kernel_CM(s, Ek0, cos0, mk)

    if jacobian == "none":
        J = np.ones_like(Ek)
    elif jacobian == "kmag":
        J = np.where(k0 > 0, kmag / k0, 0.0)
    elif jacobian == "full":
        # Jacobiano 2D por diferencias finitas centradas, con recorte de los
        # pasos de cos al rango fisico [-1, 1] (evita cos>1 en cos_lab=1).
        dE = eps * np.maximum(np.abs(Ek), 1.0)
        cp = np.minimum(cos_lab + eps, 1.0)
        cm = np.maximum(cos_lab - eps, -1.0)
        Ek0_pE, cos0_pE, _, _ = lab_to_cm(Ek + dE, cos_lab, s, mk)
        Ek0_mE, cos0_mE, _, _ = lab_to_cm(Ek - dE, cos_lab, s, mk)
        Ek0_pC, cos0_pC, _, _ = lab_to_cm(Ek, cp, s, mk)
        Ek0_mC, cos0_mC, _, _ = lab_to_cm(Ek, cm, s, mk)
        dEk0_dEk   = (Ek0_pE - Ek0_mE) / (2.0 * dE)
        dcos0_dEk  = (cos0_pE - cos0_mE) / (2.0 * dE)
        dEk0_dcos  = (Ek0_pC - Ek0_mC) / (cp - cm)
        dcos0_dcos = (cos0_pC - cos0_mC) / (cp - cm)
        J = np.abs(dEk0_dEk * dcos0_dcos - dcos0_dEk * dEk0_dcos)
    else:
        raise ValueError("jacobian debe ser 'full', 'kmag' o 'none'")

    return kern * J


if __name__ == "__main__":
    # Chequeo rapido: |F_V|^2 vs k^2. Debe dar ~1 en k^2=0 (norm. VMD) y un
    # pico enorme en k^2 ~ m_rho^2 = 0.593 (resonancia rho/omega angosta).
    print("Chequeo |F_V|^2:")
    for mk in [0.0, 0.3, 0.5, 0.77, 1.0, 1.25]:
        print(f"  mk={mk:.2f} GeV  |F_V|^2 = {np.abs(F_V(mk**2))**2:.3f}")
