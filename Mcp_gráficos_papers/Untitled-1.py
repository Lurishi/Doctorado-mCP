#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# ==========================================
# 1. PARÁMETROS FÍSICOS
# ==========================================
d = 1.5 * 1000               # Profundidad del detector en metros (1.5 km)
R_earth = 6371.0 * 1000      # Radio de la Tierra en metros (6371 km)[cite: 1]

# Parámetros de pérdida de energía[cite: 1]
a = 0.223                    # GeV/mwe[cite: 1]
b = 4.64e-4                  # mwe^-1[cite: 1]
E_f = 1.0                    # Energía final mínima en GeV[cite: 1]

# ==========================================
# 2. FUNCIÓN PARA INTEGRAR LA DENSIDAD REAL DE LA TIERRA
# ==========================================
def obtener_D_mwe_exacto(cos_th, d, R_earth):
    """
    Calcula la sobrecapa integrada en mwe usando un modelo simplificado de 
    3 capas de la Tierra (Núcleo interno/externo, Manto, Corteza).
    """
    # Posición del detector en coordenadas cartesianas (asumiendo eje Z como el cénit)
    z_det = R_earth - d
    x_det = 0.0
    
    # Dirección de la trayectoria de la partícula entrante
    # (cos_th es respecto al cénit del detector)
    sin_th = np.sqrt(1.0 - cos_th**2)
    
    # Longitud geométrica total del camino (Ecuación 3.2)[cite: 1]
    D_total = np.sqrt(z_det**2 * cos_th**2 + d * (2 * R_earth - d)) - z_det * cos_th
    
    # Si la partícula viene desde arriba (cos_th >= 0), la densidad es solo corteza superficial
    if cos_th >= 0:
        return D_total * 2.65
    
    # Para trayectorias que cruzan la Tierra (cos_th < 0), integramos numéricamente el camino
    pasos = 200
    línea_camino = np.linspace(0, D_total, pasos)
    dl = D_total / pasos
    mwe_acumulado = 0.0
    
    for l in línea_camino:
        # Calcular la distancia actual de este punto al centro de la Tierra (radio r)
        # La partícula se mueve desde la superficie hacia el detector
        dist_restante = D_total - l
        x = x_det + dist_restante * sin_th
        z = z_det + dist_restante * cos_th
        r = np.sqrt(x**2 + z**2) / 1000.0 # en km
        
        # Modelo PREM simplificado de densidades (en g/cm^3 o t/m^3)
        if r < 1221:    # Núcleo Interno
            rho = 13.0
        elif r < 3480:  # Núcleo Externo
            rho = 11.1
        elif r < 6346:  # Manto
            rho = 4.5
        else:           # Corteza
            rho = 2.65
            
        mwe_acumulado += rho * dl
        
    return mwe_acumulado

# ==========================================
# 3. ENERGÍAS INICIALES Y CONFIGURACIÓN
# ==========================================
E_initials = [2.0, 4.4, 9.7, 21.4, 47.2]
colors = ['#b83b5e', '#f9a826', '#2ca02c', '#1f77b4', '#9467bd']

cos_theta = np.linspace(-1.0, 1.0, 300)

# Calcular el vector D_mwe exacto para cada ángulo
D_mwe_vector = np.array([obtener_D_mwe_exacto(ct, d, R_earth) for ct in cos_theta])

# ==========================================
# 4. GRÁFICO
# ==========================================
plt.figure(figsize=(7, 7))

for epsilon_sq, linestyle in [(1e-4, '-'), (1e-5, '--')]: #[cite: 1]
    for E_i, color in zip(E_initials, colors):
        
        # Ecuación 3.4: Alcance R[cite: 1]
        R = (1.0 / (epsilon_sq * b)) * np.log((1.0 + (a / b) * E_i) / (1.0 + (a / b) * E_f))
        
        # Ecuación 3.3 utilizando la atenuación de masa real integrada[cite: 1]
        P = np.exp(-D_mwe_vector / R)
        
        label = f"$E_{{initial}} = {E_i}$ GeV" if epsilon_sq == 1e-4 else ""
        plt.plot(cos_theta, P, linestyle=linestyle, color=color, linewidth=1.5, label=label)

plt.xlim(-1.0, 1.0)
plt.ylim(0.1, 1.05)
plt.xlabel(r'$\cos\theta$', fontsize=12)
plt.ylabel(r'Probability to arrive with 1 [GeV]', fontsize=12)
plt.legend(loc='lower right', frameon=False, fontsize=10)
plt.text(-0.3, 0.65, r'---  $\epsilon^2 = 10^{-4}$', fontsize=11) #[cite: 1]
plt.text(-0.3, 0.60, r'- - - $\epsilon^2 = 10^{-5}$', fontsize=11) #[cite: 1]
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
# %%
