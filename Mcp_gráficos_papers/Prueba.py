import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros Físicos extraídos del paper ---
a = 0.223          # Parámetro a [GeV/mwe]
b = 4.64e-4        # Parámetro b [mwe^-1]
d_km = 1.5         # Profundidad del detector [km]
R_earth_km = 6371.0 # Radio de la Tierra [km]
E_f = 1.0          # Energía final requerida en el detector [GeV]

# Valores de energía inicial [GeV] extraídos de la leyenda de la Fig 4.
E_initials = [2.0, 4.4, 9.7, 21.4, 47.2]
colors = ['#d68f94', '#ebca71', '#54b898', '#63a3b8', '#8681b3'] # Colores aproximados del paper

# Acoplamientos epsilon^2
eps2_solid = 1e-4
eps2_dashed = 1e-5

# Eje X: coseno del ángulo cenital
cos_theta = np.linspace(-1.0, 1.0, 500)

# --- Funciones ---
def calc_D_geom(cos_t):
    """Calcula la distancia geométrica en km (Ecuación 3.2)."""
    term1 = ((R_earth_km - d_km)**2) * (cos_t**2)
    term2 = d_km * (2 * R_earth_km - d_km)
    term3 = (R_earth_km - d_km) * cos_t
    return np.sqrt(term1 + term2) - term3

def km_to_mwe(D_km, cos_t):
    """
    Aproximación del perfil de densidad terrestre.
    El paper usa el modelo estándar de la Tierra (PREM).
    Aquí usamos una densidad promedio dependiente de la trayectoria para replicar la atenuación.
    - Hacia abajo (cos_t > 0): Principalmente corteza (~2.65 g/cm^3 -> 2650 mwe/km).
    - Hacia arriba (cos_t < 0): Cruza manto/núcleo (~5.5 g/cm^3 -> 5500 mwe/km).
    """
    # Interpolación simple para emular el PREM sin importar datos externos
    density_factor = 2650.0 + 2850.0 * np.clip(-cos_t, 0, 1) 
    return D_km * density_factor

def calc_R(E_i, E_f, eps2):
    """Fórmula con el error original del paper para forzar la réplica"""
    factor1 = 1 / (eps2 * b)
    num = 1 + (a / b) * E_i  # Revertido al error del paper (a/b)
    den = 1 + (a / b) * E_f  # Revertido al error del paper (a/b)
    return factor1 * np.log(num / den)

def calc_Probability(D_mwe, R_mwe):
    """Calcula la probabilidad de supervivencia (Ecuación 3.3)."""
    return np.exp(-D_mwe / R_mwe)

# --- Generación del gráfico ---
plt.figure(figsize=(7, 6))

# Distancia para todos los ángulos
D_geom_array = calc_D_geom(cos_theta)
D_mwe_array = km_to_mwe(D_geom_array, cos_theta)

# Trazar las líneas para cada energía inicial
for E_i, col in zip(E_initials, colors):
    # Para epsilon^2 = 10^-4 (líneas continuas)
    R_mwe_4 = calc_R(E_i, E_f, eps2_solid)
    P_4 = calc_Probability(D_mwe_array, R_mwe_4)
    plt.plot(cos_theta, P_4, color=col, linestyle='-', label=f'$E_{{initial}}$ = {E_i} GeV' if eps2_solid else "")

    # Para epsilon^2 = 10^-5 (líneas punteadas)
    R_mwe_5 = calc_R(E_i, E_f, eps2_dashed)
    P_5 = calc_Probability(D_mwe_array, R_mwe_5)
    plt.plot(cos_theta, P_5, color=col, linestyle='--')

# Configuración visual del gráfico
plt.xlim(-1.0, 1.0)
plt.ylim(0.1, 1.05)
plt.xlabel(r'$\cos\theta$', fontsize=12)
plt.ylabel('Probability to arrive with 1 [GeV]', fontsize=12)

# Crear leyendas manuales para los epsilons
plt.plot([], [], color='gray', linestyle='-', label=r'$\epsilon^2=10^{-4}$')
plt.plot([], [], color='gray', linestyle='--', label=r'$\epsilon^2=10^{-5}$')

plt.legend(frameon=False, loc='lower right')
plt.minorticks_on()
plt.tick_params(direction='in', length=6, width=1, colors='k', top=True, right=True)
plt.tick_params(which='minor', direction='in', length=3, width=1, colors='k', top=True, right=True)
plt.tight_layout()

# Mostrar gráfico
plt.show()