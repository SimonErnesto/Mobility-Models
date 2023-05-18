# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
from scipy.spatial import distance_matrix
from tqdm import tqdm
import pymc as pm
import pytensor.tensor as at
import arviz as az

os.chdir(os.getcwd())

#load geolocation data and subset by Madrid
gdf = gpd.read_file("./data/zonificacion_municipios.shp")
gdf = gdf.to_crs('+proj=cea')

gdf.reset_index(inplace=True, drop=True)
gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
gdf['coords'] = [coords[0] for coords in gdf['coords']]
gdf['centroid'] = gdf["geometry"].centroid.to_crs(gdf.crs)

names = pd.read_csv("./data/nombres_municipios.csv", sep="|")
madri = names[names.name=="Madrid"].ID.values[0]

mad = gdf[gdf.ID.str.startswith("28")] #Madrid municipios start with 28

mad['area_km'] = mad.geometry.area.values/1e8

data = pd.read_csv("data_madrid_2022_monthly_ave.csv")
data['ID'] = data.origen

data = pd.merge(data,mad)

pob = pd.read_csv("./data/poblacion.csv", sep="|")
pob['ID'] = pob.municipio
pob = pob[pob.municipio.isin(data.ID.unique())]
pob = pob.drop_duplicates('municipio') 

Nj = pob[pob.ID==madri].poblacion.values 

Tdf_obs = pd.pivot_table(data[data.month=='jan'], index='origen', columns='destino', values='viajes')/31
madri_idx = Tdf_obs.columns.get_loc(madri)

data = data[data.destino==madri]
data = data[data.origen != madri]

data = pd.merge(data, pob)

Mi = data.poblacion.values

Vij_obs = data.viajes.values
Qij_obs = data.visitors.values

Ai = mad.area_km.values
Aj = mad[mad.ID==madri].area_km.values #area destination
rj = Aj/(2*np.pi)#radius destination

mad = mad[mad.ID != madri]
xy = np.array([np.array(mad.centroid.values[i].xy) for i in tqdm(range(len(mad)))]).T[0]
D = distance_matrix(xy.T, xy.T) + 0.000000001 #distance matrix in meters
R = D/1e4  #radiuses in km
rij = R[madri_idx]
rij = np.repeat(rij, 12)

month_lookup = dict(zip(data.month.unique(), range(len(data.month.unique()))))
month_id = data.month.replace(month_lookup).values #month index
months = len(pd.unique(month_id))

muni_lookup = dict(zip(data.origen.unique(), range(len(data.origen.unique()))))
muni_id = data.origen.replace(muni_lookup).values #municipio origin index
oris = len(pd.unique(muni_id))

T = 365 #monthly period

#################### Visitation Model #############################
with pm.Model() as mod_v:
    fmin = pm.Uniform("fmin", 0.1, 1, shape=(oris,months))
    fmax = pm.Uniform("fmax", 1, 50, shape=(oris,months))
    mu = pm.Wald("mu", 50, 10, shape=months)
    v = pm.Deterministic("v", (mu[month_id]*Ai)/(np.log(fmax[muni_id,month_id]/fmin[muni_id,month_id])*rij**2))
    alpha = pm.HalfNormal("alpha", 10)
    y = pm.NegativeBinomial("y", mu=v, alpha=alpha, observed=Vij_obs)

dag = pm.model_to_graphviz(mod_v)
dag.render("visitation_model_hirarchical_dag", format="png")
dag

### sample model ###
with mod_v:
    idata_v = pm.sample(1000, idata_kwargs={"log_likelihood": True})

pos_v = idata_v.stack(sample = ['chain', 'draw']).posterior
mu = pos_v['mu'].values

Vij = pos_v['v'].values*T
vij_mean = Vij.mean(axis=1)
vij_h5, vij_h95 = az.hdi(Vij.T, hdi_prob=0.9).T
Vij_vis = pd.DataFrame({"mean_v":vij_mean, "h5_v":vij_h5, "h95_v":vij_h95, "origen":data.origen.values})
Vij_vis = Vij_vis.groupby("origen", as_index=False).mean()
Vij_vis = Vij_vis.sort_values(by="mean_v")
Vij_vis_m = Vij_vis['mean_v'].values
Vij_obs_s = data.groupby("origen", as_index=False).mean()
Vij_obs_s = Vij_obs_s.sort_values(by="viajes")
Vij_obs_s = Vij_obs_s['viajes'].values
SSI_v_vis = 2*np.sum(np.minimum(Vij_obs_s, Vij_vis_m))/(np.sum(Vij_obs_s) + np.sum(Vij_vis_m))

plt.scatter(Vij_obs_s, Vij_vis_m)

Qij_pos = np.array([Ai*(T**-1)*m/(rj**2) for m in mu])  
#Qij_pos = Ai*(T**-1)*mu.T/(rj**2) 
qij_mean = Qij_pos.mean(axis=0)
qij_h5, qij_h95 = az.hdi(Qij_pos, hdi_prob=0.9).T

Qij_vis = pd.DataFrame({"mean_q":qij_mean, "h5_q":qij_h5, "h95_q":qij_h95})
Qij_vis = Qij_vis.sort_values(by="mean_q")
Qij_vis_m = Qij_vis['mean_q'].values
Qij_obs_s = np.sort(Qij_obs)
SSI_q_vis = 2*np.sum(np.minimum(Qij_obs_s, Qij_vis_m))/(np.sum(Qij_obs_s) + np.sum(Qij_vis_m))



######################## Gravity Model ################################

##Gravity model for number of trips
with pm.Model() as mod_g:
    # theta = pm.HalfNormal("theta", 2)
    omega = pm.HalfNormal("omega", 2, shape=months) 
    g_s = pm.HalfNormal("g_s", 0.25)
    gamma = pm.HalfNormal("gamma", g_s, shape=(oris,months))
    theta = pm.Gamma("theta", 0.01, 0.01, shape=months)
    lam_den = (Nj**omega[month_id])*(rij**-gamma[muni_id,month_id])
    lam_num = at.sum(lam_den)
    v = pm.Deterministic("v", theta[month_id]*Mi*(lam_den/lam_num))
    alpha = pm.HalfNormal("alpha", 3)
    y = pm.NegativeBinomial("y", mu=v, alpha=alpha, observed=Vij_obs)
    idata_g_y = pm.sample(1000, idata_kwargs={"log_likelihood": True})
pos_g_y = idata_g_y.stack(sample = ['chain', 'draw']).posterior

with pm.Model() as mod:
    # theta = pm.HalfNormal("theta", 2)
    omega = pm.HalfNormal("omega", 2, shape=months) 
    g_s = pm.HalfNormal("g_s", 0.25)
    gamma = pm.HalfNormal("gamma", g_s, shape=(oris,months))
    theta = pm.Gamma("theta", 0.01, 0.01, shape=months)
    lam_den = (Nj**omega[month_id])*(rij**-gamma[muni_id,month_id])
    lam_num = at.sum(lam_den)
    v = pm.Deterministic("v", theta[month_id]*Mi*(lam_den/lam_num))
    alpha = pm.HalfNormal("alpha", 3)
    w = pm.NegativeBinomial("w", mu=v, alpha=alpha, observed=Qij_obs)
    idata_g_w = pm.sample(1000, idata_kwargs={"log_likelihood": True})
pos_g_w = idata_g_w.stack(sample = ['chain', 'draw']).posterior  

Vij = pos_g_y['v'].values
vij_mean = Vij.mean(axis=1)
vij_h5, vij_h95 = az.hdi(Vij.T, hdi_prob=0.9).T
Vij_gra = pd.DataFrame({"mean_v":vij_mean, "h5_v":vij_h5, "h95_v":vij_h95, "origen":data.origen.values})
Vij_gra = Vij_gra.groupby("origen", as_index=False).mean()
Vij_gra = Vij_gra.sort_values(by="mean_v")
Vij_gra_m = Vij_gra['mean_v'].values
Vij_obs_s = data.groupby("origen", as_index=False).mean()
Vij_obs_s = Vij_obs_s.sort_values(by="viajes")
Vij_obs_s = Vij_obs_s['viajes'].values
SSI_v_vis = 2*np.sum(np.minimum(Vij_obs_s, Vij_gra_m))/(np.sum(Vij_obs_s) + np.sum(Vij_gra_m))

plt.scatter(Vij_obs_s, Vij_gra_m)


Qij = pos_g_w['v'].values
qij_mean = Qij.mean(axis=1)
qij_h5, qij_h95 = az.hdi(Qij.T, hdi_prob=0.9).T
Qij_gra = pd.DataFrame({"mean_q":qij_mean, "h5_q":qij_h5, "h95_q":qij_h95, "origen":data.origen.values})
Qij_gra = Qij_gra.groupby("origen", as_index=False).mean()
Qij_gra = Qij_gra.sort_values(by="mean_q")
Qij_gra_m = Qij_gra['mean_q'].values
Qij_obs_s = data.groupby("origen", as_index=False).mean()
Qij_obs_s = Qij_obs_s.sort_values(by="visitors")
Qij_obs_s = Qij_obs_s['visitors'].values
SSI_q_vis = 2*np.sum(np.minimum(Qij_obs_s, Qij_gra_m))/(np.sum(Qij_obs_s) + np.sum(Qij_gra_m))

plt.scatter(Qij_obs_s, Qij_gra_m)