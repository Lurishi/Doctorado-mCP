#%%
import pandas as pd
import numpy as np

#%%

df = pd.read_csv("/home/lurishi/Escritorio/Doctorado/Heavenly_mCP/MCP_flux/MCP_AllMesons_SURFACE_sum.csv")

# Cambia estos nombres por las columnas que quieras eliminar
df = df.drop(columns=['flux_pi0', 'flux_eta', 'flux_rho', 'flux_omega', 'flux_phi', 'flux_jpsi'])

df.to_csv("MCP_total_flux_surface.csv", index=False)

# %%
