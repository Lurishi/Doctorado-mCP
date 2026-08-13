#--------------------------------------------------------------------------------------------------------------------------------------
#  This script sums the (unattenuated) MCP flux contributions from every parent meson
#  (pi0, eta, rho, omega, phi, J/psi) evaluated "at the surface", for each combination
#  of (mass, eps2, cos theta), and writes out the combined flux vs energy.
#--------------------------------------------------------------------------------------------------------------------------------------
#%%
import pandas as pd
import numpy as np
from scipy import interpolate
import os
#%%
cwd = os.getcwd()
print('Sum_MCP_flux_mesons.py called!')

#--- Path where the individual meson flux files live (same convention as Rate_for_cosine.py)
path = '/home/lurishi/Escritorio/HeavenlyMCP/Data/MCP_flux/MCP_flux'

#--- Names -> file names of each parent-meson contribution
MESON_FILES = {
    'pi0':   'MCP_FROM_PI0_SURFACE.csv',
    'eta':   'MCP_FROM_ETA_SURFACE.csv',
    'rho':   'MCP_FROM_RHO_SURFACE.csv',
    'omega': 'MCP_FROM_OMEGA_SURFACE.csv',
    'phi':   'MCP_FROM_PHI_SURFACE.csv',
    'jpsi':  'MCP_FROM_JPSI_SURFACE.csv',
}


def load_meson_fluxes(path=path, files=MESON_FILES):
    """
    Reads every parent-meson flux file and standardizes the column names to:
    ['flux', 'e', 'cos', 'mass', 'eps2']
    Returns a dict {meson_name: dataframe}.
    """
    dfs = {}
    for name, fname in files.items():
        fpath = os.path.join(path, fname)
        d = pd.read_csv(fpath)
        d = d.rename(columns={
            'dphi_dEdCth(1/GeV/s/cm2)': 'flux',
            'Energy (GeV)': 'e',
            'cosTh': 'cos',
            'm(GeV)': 'mass',
        })
        dfs[name] = d.sort_values('e').reset_index(drop=True)
    return dfs


def sum_fluxes_for_combo(dfs, mass_v, eps2_v, cos_v):
    """
    For a fixed (mass, eps2, cos) combination, builds a common energy grid
    (the union of the energy points of every meson that has data for this
    combination) and sums the interpolated flux of each meson on that grid.
    Mesons with no data for this combination simply contribute 0.

    Returns a dataframe with columns: e, flux_total, flux_<meson> (per meson).
    """
    subsets = {}
    for name, d in dfs.items():
        sub = d[(np.isclose(d['mass'], mass_v)) &
                (np.isclose(d['eps2'], eps2_v)) &
                (np.isclose(d['cos'], cos_v))]
        if len(sub) > 0:
            subsets[name] = sub.sort_values('e').reset_index(drop=True)

    if len(subsets) == 0:
        return pd.DataFrame(columns=['e', 'flux_total'])

    # Common energy grid = union of all energies available for this combo
    e_grid = np.unique(np.concatenate([sub['e'].values for sub in subsets.values()]))

    out = pd.DataFrame({'e': e_grid})
    out['flux_total'] = 0.0

    for name, sub in subsets.items():
        emin, emax = sub['e'].min(), sub['e'].max()
        interp = interpolate.interp1d(sub['e'], sub['flux'], bounds_error=False, fill_value=0.0)
        flux_interp = interp(e_grid)
        # Outside [emin, emax] there's no meson-flux info -> contributes 0
        flux_interp = np.where((e_grid >= emin) & (e_grid <= emax), flux_interp, 0.0)
        out[f'flux_{name}'] = flux_interp
        out['flux_total'] += flux_interp

    out['mass'] = mass_v
    out['eps2'] = eps2_v
    out['cos'] = cos_v
    return out


def sum_all_mesons(path=path, files=MESON_FILES):
    """
    Loops over every (mass, eps2, cos) combination present in the meson files
    and sums the contributions of all parent mesons.
    Returns one combined dataframe with columns:
    ['cos', 'eps2', 'mass', 'e', 'flux_total', 'flux_pi0', 'flux_eta', ...]
    """
    dfs = load_meson_fluxes(path, files)

    # Union of all (mass, eps2, cos) combinations across every meson file
    combos = pd.concat([d[['mass', 'eps2', 'cos']] for d in dfs.values()]).drop_duplicates()

    results = []
    for _, row in combos.iterrows():
        res = sum_fluxes_for_combo(dfs, row['mass'], row['eps2'], row['cos'])
        if len(res) > 0:
            results.append(res)

    df_sum = pd.concat(results, ignore_index=True)
    # Reorder columns nicely
    front = ['cos', 'eps2', 'mass', 'e', 'flux_total']
    other = [c for c in df_sum.columns if c not in front]
    df_sum = df_sum[front + other].sort_values(['mass', 'eps2', 'cos', 'e']).reset_index(drop=True)
    return df_sum


if __name__ == '__main__':
    df_sum = sum_all_mesons()
    print('Combined dataframe shape:', df_sum.shape)
    print(df_sum.head())

    # Save the combined dataframe to a CSV file
    out_path = path + '/MCP_AllMesons_SURFACE_sum.csv'
    df_sum.to_csv(out_path, index=False)
    print('Saved to', out_path)
# %%
