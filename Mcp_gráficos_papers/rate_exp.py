#%%
'''
INFO:
- Este script contiene una funcion que calcula el rate de eventos de ne electrones
esperados correspondiente al upper limit con un nivel de confianza dado
'''
#%%
import numpy as np
from scipy.stats import chi2
def rate_exp(image,ne,tau):
    '''
    Necesita:
    - numpy
    - scipy.stats.chi2
    ---------------
    parámetros:
    image: np.ndarray
    - imagen 2D leida en electrones
    ne: float
    - número de electrones
    tau: float
    - tiempo de exposición en días

    ---------------
    return:
    rate: float
    - tasa de eventos por kg por día (kg^-1 day^-1)
    mu_ul: float
    - límite superior de mu al 90% de confianza
    neventos: int
    - número de eventos de "ne" electrones observados

    '''
    masa_CCD =  3.6e-7*(np.shape(image)[0]*np.shape(image)[1]) #masa en kg
    

    if ne ==1:
        eventos_n_e = image[np.logical_and(image>0.68,image<ne+0.5)]
    else:
        eventos_n_e = image[np.logical_and(image>ne-0.5,image<ne+0.5)]
    neventos = len(eventos_n_e)
    print(neventos)
    p_on_the_left = 1 - 0.9
    degrees_of_freedom = 2 * (neventos + 1)
    x = chi2.ppf(1 - p_on_the_left, degrees_of_freedom)
    mu_ul = x / 2

    rate = mu_ul/((tau+2.5)*masa_CCD) # tiempo tau en días, el tiempo expuesto es 60 hs
    return rate, mu_ul, neventos
#%%
def rate_exp_neventos(neventos,tau,img_size):
    '''
    Necesita:
    - numpy
    - scipy.stats.chi2
    ---------------
    parámetros:
    neventos: int
    - número de eventos de "ne" electrones observados
    tau: float
    - tiempo de exposición en días
    img_size: tuple
    - tamaño de la imagen (pixeles) (alto, ancho)

    ---------------
    return:
    rate: float
    - tasa de eventos por kg por día (kg^-1 day^-1)
    mu_ul: float
    - límite superior de mu al 90% de confianza

    '''
    masa_CCD =  3.6e-7*(img_size[0]*img_size[1]) #masa en kg
    p_on_the_left = 1 - 0.9
    degrees_of_freedom = 2 * (neventos + 1)
    x = chi2.ppf(1 - p_on_the_left, degrees_of_freedom)
    mu_ul = x / 2

    rate = mu_ul/((tau+2.5)*masa_CCD) # tiempo tau en días, el tiempo expuesto es 60 hs
    return rate, mu_ul
#%%

def limites_exclusion(r_exp, df, sigma,ne):
   if ne < 1 or ne > 10:
       raise ValueError("ne debe estar entre 1 y 10")
   if not isinstance(ne, int):
       raise TypeError("ne debe ser un entero")
   if sigma <= 0:
       raise ValueError("La sección eficaz del dataframe debe ser un número positivo")
  
   r_teo = np.array(df[f'ne{ne}'])
   sigma_exp = (r_exp/r_teo)*sigma
  
   return sigma_exp  