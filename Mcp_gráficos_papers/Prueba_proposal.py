#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# --- 1. Configuración de parámetros físicos ---
L0 = 1e5    # Alcance de referencia (ajustable según el paper)
eps2 = 1e-4 # Acoplamiento (factor epsilon^2)
R_earth = 6371.0 # Radio de la Tierra en km

# --- 2. Densidad variable tipo PREM ---
def get_density(r):
    # r es el radio desde el centro en km
    if r > 6346: return 2.6   # Corteza[cite: 1]
    if r > 3480: return 3.9   # Manto[cite: 1]
    return 9.9                # Núcleo[cite: 1]

def get_average_density_rad(theta_rad):
    # Calcula la densidad promedio a lo largo de la trayectoria
    # Trayectoria definida por theta entre -pi y 0
    D = 2 * R_earth * np.abs(np.cos(theta_rad))
    
    # Muestreo de la densidad a lo largo de la cuerda
    # La profundidad radial r depende de la distancia recorrida 's'
    s_values = np.linspace(0, D, 50)
    # r^2 = R^2 + s^2 - 2Rs*cos(pi - theta)
    r_values = np.sqrt(R_earth**2 + s_values**2 - 2*R_earth*s_values*np.cos(np.pi - np.abs(theta_rad)))
    
    densities = [get_density(r) for r in r_values]
    return np.mean(densities)

# --- 3. Función de Probabilidad ---
def survival_prob_rad(theta_rad):
    # Distancia recorrida en la Tierra
    D = 2 * R_earth * np.abs(np.cos(theta_rad))
    # Densidad media ponderada
    rho_avg = get_average_density_rad(theta_rad)
    
    # Alcance medio ajustado por densidad y carga
    R_avg = L0 / (rho_avg * eps2)
    
    # Probabilidad exponencial de supervivencia
    return np.exp(-D / R_avg)

# --- 4. Generación y Graficado ---
# Rango solicitado: -pi a 0
angles_rad = np.linspace(-np.pi, 0, 100)
probs = [survival_prob_rad(a) for a in angles_rad]

plt.figure(figsize=(8, 5))
plt.plot(np.cos(angles_rad), probs, label='Probabilidad de Supervivencia')
plt.xlabel(r'$\cos(\theta)$')
plt.ylabel('Probabilidad de Supervivencia')
plt.title('Atenuación de mCP en la Tierra (Modelo PREM simplificado)')
plt.grid(True)
plt.legend()
plt.show()
# %%
