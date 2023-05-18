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

data = pd.read_csv("data_jan_ave.csv")

Tdf_obs = pd.pivot_table(data, index='origen', columns='destino', values='viajes')/31
madri_idx = Tdf_obs.columns.get_loc(madri)

data = data[data.destino==madri] #data.groupby(["origen", "destino"], as_index=False).mean()
data['ID'] = data.origen

pob = pd.read_csv("./data/poblacion.csv", sep="|")
pob['ID'] = pob.municipio
pob = pob[pob.municipio.isin(data.ID.unique())]
pob = pob.drop_duplicates('municipio') 

data = pd.merge(data, pob)

#################### Observed Data ##############################
### Plot Survey data observed
#Tdf_obs = data.groupby(['origen', 'destino']).mean()

Ndf = pd.DataFrame({"ID":Tdf_obs[madri].index, madri:Tdf_obs[madri].values})
Ndf = Ndf.replace(np.nan,0)
madg = pd.merge(mad,Ndf)
madg = madg.sort_values(by=madri, ascending=False)
madg.reset_index(inplace=True, drop=True)

max_val = int(max(madg[madri]))
mid_val = int(np.median(madg[madri]))
min_val = int(min(madg[madri]))

colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

ch_map = mad.plot(figsize=(10,20), color="tan", alpha=0.2, edgecolor="k")
for c in range(len(madg)):
    #ch_map.set_facecolor("grey")
    muni_dest = madg.ID[c]
    orig = madri
    dest_coord = mad[mad.ID==muni_dest].coords.values[0]
    orig_coord = mad[mad.ID==orig].coords.values[0]
    x = (dest_coord[0], orig_coord[0])
    y = (dest_coord[1], orig_coord[1])
    ch_map.plot(x,y, color=colors[c])  
cbar = plt.colorbar(sm,fraction=0.005, pad=0.001)
cbar.set_ticklabels([min_val,mid_val,max_val])
plt.title("Observed Data: Average Daily Travellers to Central Madrid Jan 2022", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_to_central_madrid_observed.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

#data = data[data.origen != madri]
#mad = mad[mad.ID != madri]

#create distance matrix for radiuses R
xy = np.array([np.array(mad.centroid.values[i].xy) for i in tqdm(range(len(mad)))]).T[0]
D = distance_matrix(xy.T, xy.T) #+ 0.000000001 #distance matrix in meters
R = D/1e4  #radiuses in km
rij = R[madri_idx]
rij = rij[rij != rij[madri_idx]]

Ai = 2*np.pi*rij #area between origins and destination
#Ai = gdf[gdf.ID.isin(data.origen.unique())].geometry.area/1e8 #area of origins
#Ai = Ai.values
#Ai = Ai[Ai != Ai[madri_idx]]

Aj = gdf[gdf.ID==madri].geometry.area/1e8 #area destination
Aj = Aj.values

Nj = pob[pob.municipio==madri].poblacion.values 
Nj = Nj[0]

rj = Aj/(2*np.pi)
rj = rj[0] #radius destination
rhoj = Nj/Aj


fmin = 1/31 #data.viajes/data.visitors/31 #minimum frequency
fmax = 1 #data.viajes/data.visitors

Njv = data.visitors.values #visitors destination from origins

Mi = data.poblacion.values
Mi = Mi[Mi != Mi[madri_idx]]

Vij_obs = data.viajes.values
Vij_obs = Vij_obs[Vij_obs != Vij_obs[madri_idx]]

Qij_obs = data.visitors.values
Qij_obs = Qij_obs[Qij_obs != Qij_obs[madri_idx]]

T = 31 #length of period under study

#################### Visitation Model #############################
with pm.Model() as mod_v:
    fmin = pm.Uniform("fmin", 0.1, 1, shape=len(Vij_obs))
    fmax = pm.Uniform("fmax", 1, 50, shape=len(Vij_obs))
    mu = pm.Wald("mu", 60, 10) #pm.Uniform("mu", 1e4, 1e5) #, shape=len(Vij_obs))
    v = pm.Deterministic("v", (mu*Ai)/(np.log(fmax/fmin)*rij**2))
    alpha = pm.HalfNormal("alpha", 10)
    y = pm.NegativeBinomial("y", mu=v, alpha=alpha, observed=Vij_obs)

dag = pm.model_to_graphviz(mod_v)
dag.render("visitation_model_dag", format="png")
dag

### sample model ###
with mod_v:
    idata_v_y = pm.sample(1000, idata_kwargs={"log_likelihood": True})

pos_v_y = idata_v_y.stack(sample = ['chain', 'draw']).posterior
mu = pos_v_y['mu'].values

Vij = pos_v_y['v'].values#*T
#Vij = Vij[Vij.mean(axis=1) != max(Vij.mean(axis=1))]
vij_mean = Vij.mean(axis=1)
vij_h5, vij_h95 = az.hdi(Vij.T, hdi_prob=0.9).T
Vij_vis = pd.DataFrame({"mean_v":vij_mean, "h5_v":vij_h5, "h95_v":vij_h95})
Vij_vis = Vij_vis.sort_values(by="mean_v")
Vij_vis_m = Vij_vis['mean_v'].values
#Vij_obs_s = np.sort(Vij_obs[Vij_obs != max(Vij_obs)])
Vij_obs_s = np.sort(Vij_obs)
SSI_v_vis = 2*np.sum(np.minimum(Vij_obs_s, Vij_vis_m))/(np.sum(Vij_obs_s) + np.sum(Vij_vis_m))


with pm.Model() as mod_q:
    fmin = pm.Uniform("fmin", 0.1, 1, shape=len(Qij_obs))
    mu = pm.Wald("mu", 50, 10) 
    q = pm.Deterministic("q", Ai*(mu/(fmin*rj**2)))
    alpha = pm.HalfNormal("alpha", 10)
    w = pm.NegativeBinomial("w", mu=q, alpha=alpha, observed=Qij_obs)

dag = pm.model_to_graphviz(mod_v)
dag.render("visitation_model_dag", format="png")
dag

### sample model ###
with mod_q:
    idata_v_w = pm.sample(1000, idata_kwargs={"log_likelihood": True})

pos_v_w = idata_v_w.stack(sample = ['chain', 'draw']).posterior
mu = pos_v_w['mu'].values
 
Qij = pos_v_w['q'].values
qij_mean = Qij.mean(axis=1)
qij_h5, qij_h95 = az.hdi(Qij.T, hdi_prob=0.9).T
Qij_vis = pd.DataFrame({"mean_q":qij_mean, "h5_q":qij_h5, "h95_q":qij_h95})
Qij_vis = Qij_vis.sort_values(by="mean_q")
Qij_vis_m = Qij_vis['mean_q'].values
Qij_obs_s = np.sort(Qij_obs)
SSI_q_vis = 2*np.sum(np.minimum(Qij_obs_s, Qij_vis_m))/(np.sum(Qij_obs_s) + np.sum(Qij_vis_m))



######################## Radiation Model ################################

Sij = []
for i in tqdm(range(rij.shape[0])):
    s = rij<=rij[i] #exclude places outside radius
    # calculate populations sum within radious excluding origin and destination pops 
    p = Mi[s].sum() - Mi[i] 
    Sij.append(p)
Sij = np.array(Sij) 

## Radiation model for number of trips
with pm.Model() as mod_rv:
    #Ti = pm.Uniform("Ti", 1e3, 1e5, shape=len(Vij_obs)) 
    Ti = pm.Wald("Ti", 600000, 100000, shape=len(Vij_obs)) #pm.Exponential("Ti", 0.00001, shape=len(Vij_obs))
    p_num = Mi*Nj
    p_den = (Mi + Sij)*(Mi + Nj + Sij)
    v = pm.Deterministic("v", Ti*(p_num/p_den))
    alpha = pm.HalfNormal("alpha", 10)
    y = pm.NegativeBinomial("y", mu=v, alpha=alpha, observed=Vij_obs)

dag = pm.model_to_graphviz(mod_rv)
dag.render("radiation_model_dag", format="png")
dag

### sample model ###
with mod_rv:
    idata_r_y = pm.sample(1000, idata_kwargs={"log_likelihood": True})

pos_r_y = idata_r_y.stack(sample = ['chain', 'draw']).posterior


with pm.Model() as mod_rq:
    #Ti = pm.Uniform("Ti", 1e4, 1e5, shape=len(Vij_obs)) 
    Ti = pm.Wald("Ti", 500000, 100000, shape=len(Vij_obs)) #pm.Exponential("Ti", 0.0001, shape=len(Vij_obs))
    p_num = Mi*Nj
    p_den = (Mi + Sij)*(Mi + Nj + Sij)
    v = pm.Deterministic("q", Ti*(p_num/p_den))
    alpha = pm.HalfNormal("alpha", 10)
    w = pm.NegativeBinomial("w", mu=v, alpha=alpha, observed=Qij_obs)
    idata_r_w = pm.sample(1000, idata_kwargs={"log_likelihood": True})
pos_r_w = idata_r_w.stack(sample = ['chain', 'draw']).posterior


Vij = pos_r_y['v'].values*T
#Vij = Vij[Vij.mean(axis=1) != max(Vij.mean(axis=1))]
vij_mean = Vij.mean(axis=1)
vij_h5, vij_h95 = az.hdi(Vij.T, hdi_prob=0.9).T
Vij_rad = pd.DataFrame({"mean_v":vij_mean, "h5_v":vij_h5, "h95_v":vij_h95})
Vij_rad = Vij_rad.sort_values(by="mean_v")
Vij_rad_m = Vij_rad['mean_v'].values
#Vij_obs_s = np.sort(Vij_obs)
SSI_v_rad = 2*np.sum(np.minimum(Vij_obs_s, Vij_rad_m))/(np.sum(Vij_obs_s) + np.sum(Vij_rad_m))

Qij_pos = pos_r_w['q'].values#*T #pos_v['q'].values.mean(axis=1)
qij_mean = Qij_pos.mean(axis=1)
qij_h5, qij_h95 = az.hdi(Qij_pos.T, hdi_prob=0.9).T

Qij_rad = pd.DataFrame({"mean_q":qij_mean, "h5_q":qij_h5, "h95_q":qij_h95})
Qij_rad = Qij_rad.sort_values(by="mean_q")
Qij_rad_m = Qij_rad['mean_q'].values
Qij_obs_s = np.sort(Qij_obs)
SSI_q_rad = 2*np.sum(np.minimum(Qij_obs_s, Qij_rad_m))/(np.sum(Qij_obs_s) + np.sum(Qij_rad_m))



######################## Gravity Model ################################
##Gravity model for number of trips
with pm.Model() as mod_gv:
    omega = pm.Gamma("omega", 1, 1) 
    gamma = pm.Gamma("gamma", 1, 1, shape=len(Qij_obs)) 
    theta = pm.Gamma("theta", 0.001, 0.001)
    lam_den = (Nj**omega)*(rij**-gamma)
    lam_num = at.sum(lam_den)
    v = pm.Deterministic("v", theta*Mi*(lam_den/lam_num))
    alpha = pm.HalfNormal("alpha", 10)
    y = pm.NegativeBinomial("y", mu=v, alpha=alpha, observed=Vij_obs)
    idata_g_y = pm.sample(1000, idata_kwargs={"log_likelihood": True})
pos_g_y = idata_g_y.stack(sample = ['chain', 'draw']).posterior

with pm.Model() as mod_gq:
    omega = pm.Gamma("omega", 1, 1) 
    gamma = pm.Gamma("gamma", 1, 1, shape=len(Qij_obs)) 
    theta = pm.Gamma("theta", 0.001, 0.001)
    lam_den = (Nj**omega)*(rij**-gamma)
    lam_num = at.sum(lam_den)
    v = pm.Deterministic("q", theta*Mi*(lam_den/lam_num))
    alpha = pm.HalfNormal("alpha", 10)
    w = pm.NegativeBinomial("w", mu=v, alpha=alpha, observed=Qij_obs)
    idata_g_w = pm.sample(1000, idata_kwargs={"log_likelihood": True})
pos_g_w = idata_g_w.stack(sample = ['chain', 'draw']).posterior  

Vij = pos_g_y['v'].values#*T
#Vij = Vij[Vij.mean(axis=1) != max(Vij.mean(axis=1))]
vij_mean = Vij.mean(axis=1)
vij_h5, vij_h95 = az.hdi(Vij.T, hdi_prob=0.9).T
Vij_gra = pd.DataFrame({"mean_v":vij_mean, "h5_v":vij_h5, "h95_v":vij_h95})
Vij_gra = Vij_gra.sort_values(by="mean_v")
Vij_gra_m = Vij_gra['mean_v'].values
#Vij_obs_s = np.sort(Vij_obs)
SSI_v_gra = 2*np.sum(np.minimum(Vij_obs_s, Vij_gra_m))/(np.sum(Vij_obs_s) + np.sum(Vij_gra_m))

Qij_pos = pos_g_w['q'].values#*T #pos_v['q'].values.mean(axis=1)
qij_mean = Qij_pos.mean(axis=1)
qij_h5, qij_h95 = az.hdi(Qij_pos.T, hdi_prob=0.9).T
Qij_gra = pd.DataFrame({"mean_q":qij_mean, "h5_q":qij_h5, "h95_q":qij_h95})
Qij_gra = Qij_gra.sort_values(by="mean_q")
Qij_gra_m = Qij_gra['mean_q'].values
Qij_obs_s = np.sort(Qij_obs)
SSI_q_gra = 2*np.sum(np.minimum(Qij_obs_s, Qij_gra_m))/(np.sum(Qij_obs_s) + np.sum(Qij_gra_m))



######################## Plot Models Comparison ###############################

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))
ax[0,0].plot(Vij_obs_s, Vij_obs_s, color='k', linestyle=":", label="Observed")
ax[0,0].scatter(Vij_obs_s, Vij_vis_m, marker="o", facecolor="w", color="g", label="Visitation: SSI="+str(round(SSI_v_vis, 2)))
ax[0,0].scatter(Vij_obs_s, Vij_rad_m, marker="^", facecolor="w", color="b", label="Radiation: SSI="+str(round(SSI_v_rad, 2)))
ax[0,0].scatter(Vij_obs_s, Vij_gra_m, marker="x", color="r", label="Gravity: SSI="+str(round(SSI_v_gra, 2)))
ax[0,0].legend()
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("Average Daily Trips")
ax[0,0].set_ylabel("Estmiate")
ax[0,0].set_xlabel("Observed")

ax[0,1].plot(Vij_obs_s[:85], Vij_obs_s[:85], color='k', linestyle=":", label="Observed")
SSI_v_vis2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_vis_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_vis_m[:85]))
SSI_v_rad2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_rad_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_rad_m[:85]))
SSI_v_gra2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_gra_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_gra_m[:85]))
ax[0,1].scatter(Vij_obs_s[:85], Vij_vis_m[:85], marker="o", facecolor="w", color="g", label="Visitation: SSI="+str(round(SSI_v_vis2, 2)))
ax[0,1].scatter(Vij_obs_s[:85], Vij_rad_m[:85], marker="^", facecolor="w", color="b", label="Radiation: SSI="+str(round(SSI_v_rad2, 2)))
ax[0,1].scatter(Vij_obs_s[:85], Vij_gra_m[:85], marker="x", color="r", label="Gravity: SSI="+str(round(SSI_v_gra2, 2)))
ax[0,1].legend()
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("Average Daily Trips < "+str(round(max(Vij_obs_s[:85]))))
ax[0,1].set_ylabel("Estmiate")
ax[0,1].set_xlabel("Observed")

ax[1,0].plot(Qij_obs_s, Qij_obs_s, color='k', linestyle=":", label="Observed")
ax[1,0].scatter(Qij_obs_s, Qij_vis_m, marker="o", facecolor="w", color="g", label="Visitation: SSI="+str(round(SSI_q_vis, 2)))
ax[1,0].scatter(Qij_obs_s, Qij_rad_m, marker="^", facecolor="w", color="b", label="Radiation: SSI="+str(round(SSI_q_rad, 2)))
ax[1,0].scatter(Qij_obs_s, Qij_gra_m, marker="x", color="r", label="Gravity: SSI="+str(round(SSI_q_gra, 2)))
ax[1,0].legend()
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("Average Daily Visitors")
ax[1,0].set_ylabel("Estmiate")
ax[1,0].set_xlabel("Observed")

ax[1,1].plot(Qij_obs_s[:87], Qij_obs_s[:87], color='k', linestyle=":", label="Observed")
SSI_q_vis2 = 2*np.sum(np.minimum(Qij_obs_s[:87], Qij_vis_m[:87]))/(np.sum(Qij_obs_s[:87]) + np.sum(Qij_vis_m[:87]))
SSI_q_gra2 = 2*np.sum(np.minimum(Qij_obs_s[:87], Qij_gra_m[:87]))/(np.sum(Qij_obs_s[:87]) + np.sum(Qij_gra_m[:87]))
SSI_q_rad2 = 2*np.sum(np.minimum(Qij_obs_s[:87], Qij_rad_m[:87]))/(np.sum(Qij_obs_s[:87]) + np.sum(Qij_rad_m[:87]))
ax[1,1].scatter(Qij_obs_s[:87], Qij_vis_m[:87], marker="o", facecolor="w", color="g", label="Visitation: SSI="+str(round(SSI_q_vis2, 2)))
ax[1,1].scatter(Qij_obs_s[:87], Qij_rad_m[:87], marker="^", facecolor="w", color="b", label="Radiation: SSI="+str(round(SSI_q_rad2, 2)))
ax[1,1].scatter(Qij_obs_s[:87], Qij_gra_m[:87], marker="x", color="r", label="Gravity: SSI="+str(round(SSI_q_gra2, 2)))
ax[1,1].grid(alpha=0.2)
ax[1,1].legend()
ax[1,1].set_title("Average Daily Visitors < "+str(round(max(Qij_obs_s[:85]))))
ax[1,1].set_ylabel("Estmiate")
ax[1,1].set_xlabel("Observed")

plt.suptitle("Bayesian Visitation Model: Trips and Travellers to Central Madrid", fontsize=18)
plt.tight_layout()
plt.savefig("bayesian_visitation_model_estimates.png", dpi=300)
plt.show()
plt.close()



fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))

ax[0,0].plot(np.arange(len(Vij_obs_s)), Vij_obs_s, color='k', linestyle=":", label="Observed")
ax[0,0].plot(np.arange(len(Vij_obs_s)), Vij_vis.mean_v, color="g", label="Visitation: SSI="+str(round(SSI_v_vis, 2)))
ax[0,0].fill_between(np.arange(len(Vij_obs_s)), Vij_vis.h5_v, Vij_vis.h95_v, color="g", alpha=0.2)
ax[0,0].plot(np.arange(len(Vij_obs_s)), Vij_gra.mean_v, linestyle="--", color="orangered", label="Gravity: SSI="+str(round(SSI_v_gra, 2)))
ax[0,0].fill_between(np.arange(len(Vij_obs_s)), Vij_gra.h5_v, Vij_gra.h95_v, color="orangered", alpha=0.2)
ax[0,0].legend()
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("Average Daily Trips")
ax[0,0].set_ylabel("Estmiate")
ax[0,0].set_xlabel("Observed")

SSI_v_vis2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_vis_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_vis_m[:85]))
SSI_v_gra2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_gra_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_gra_m[:85]))
ax[0,1].plot(np.arange(len(Vij_obs_s))[:85], Vij_obs_s[:85], color='k', linestyle=":", label="Observed")
ax[0,1].plot(np.arange(len(Vij_obs_s))[:85], Vij_vis.mean_v[:85], color="g", label="Visitation: SSI="+str(round(SSI_v_vis2, 2)))
ax[0,1].fill_between(np.arange(len(Vij_obs_s))[:85], Vij_vis.h5_v[:85], Vij_vis.h95_v[:85], color="g", alpha=0.2)
ax[0,1].plot(np.arange(len(Vij_obs_s))[:85], Vij_gra.mean_v[:85], linestyle="--", color="orangered", label="Gravity: SSI="+str(round(SSI_v_gra2, 2)))
ax[0,1].fill_between(np.arange(len(Vij_obs_s))[:85], Vij_gra.h5_v[:85], Vij_gra.h95_v[:85], color="orangered", alpha=0.2)
ax[0,1].legend()
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("Average Daily Trips < "+str(round(max(Vij_obs_s[:85]))))
ax[0,1].set_ylabel("Estmiate")
ax[0,1].set_xlabel("Observed")

ax[1,0].plot(np.arange(len(Qij_obs_s)), Qij_obs_s, color='k', linestyle=":", label="Observed")
ax[1,0].plot(np.arange(len(Qij_obs_s)), Qij_vis.mean_q, color="g", label="Visitation: SSI="+str(round(SSI_q_vis, 2)))
ax[1,0].fill_between(np.arange(len(Qij_obs_s)), Qij_vis.h5_q, Qij_vis.h95_q, color="g", alpha=0.2)
ax[1,0].plot(np.arange(len(Qij_obs_s)), Qij_gra.mean_q, linestyle="--", color="orangered", label="Gravity: SSI="+str(round(SSI_q_gra, 2)))
ax[1,0].fill_between(np.arange(len(Qij_obs_s)), Qij_gra.h5_q, Qij_gra.h95_q, color="orangered", alpha=0.2)
ax[1,0].legend()
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("Average Daily Travellers")
ax[1,0].set_ylabel("Estmiate")
ax[1,0].set_xlabel("Observed")

SSI_q_gra2 = 2*np.sum(np.minimum(Qij_obs_s[:85], Qij_gra_m[:85]))/(np.sum(Qij_obs_s[:85]) + np.sum(Qij_gra_m[:85]))
SSI_q_vis2 = 2*np.sum(np.minimum(Qij_obs_s[:85], Qij_vis_m[:85]))/(np.sum(Qij_obs_s[:85]) + np.sum(Qij_vis_m[:85]))
ax[1,1].plot(np.arange(len(Qij_obs_s))[:85], Qij_obs_s[:85], color='k', linestyle=":", label="Observed")
ax[1,1].plot(np.arange(len(Qij_obs_s))[:85], Qij_vis.mean_q[:85], color="g", label="Visitation: SSI="+str(round(SSI_q_vis2, 2)))
ax[1,1].fill_between(np.arange(len(Qij_obs_s))[:85], Qij_vis.h5_q[:85], Qij_vis.h95_q[:85], color="g", alpha=0.2)
ax[1,1].plot(np.arange(len(Qij_obs_s))[:85], Qij_gra.mean_q[:85], linestyle="--", color="orangered", label="Gravity: SSI="+str(round(SSI_q_gra2, 2)))
ax[1,1].fill_between(np.arange(len(Qij_obs_s))[:85], Qij_gra.h5_q[:85], Qij_gra.h95_q[:85], color="orangered", alpha=0.2)
ax[1,1].legend()
ax[1,1].grid(alpha=0.2)
ax[1,1].set_title("Average Daily Travellers < "+str(round(max(Qij_obs_s[:85]))))
ax[1,1].set_ylabel("Estmiate")
ax[1,1].set_xlabel("Observed")

plt.suptitle("Visitation vs Gravity: Trips and Travellers to Central Madrid", fontsize=18)
plt.tight_layout()
plt.savefig("vistation_gravity_comparison.png", dpi=300)
plt.show()
plt.close()





fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))

ax[0,0].plot(np.arange(len(Vij_obs_s)), Vij_obs_s, color='k', linestyle=":", label="Observed")
ax[0,0].plot(np.arange(len(Vij_obs_s)), Vij_vis.mean_v, color="g", label="Visitation: SSI="+str(round(SSI_v_vis, 2)))
ax[0,0].fill_between(np.arange(len(Vij_obs_s)), Vij_vis.h5_v, Vij_vis.h95_v, color="g", alpha=0.2)
ax[0,0].plot(np.arange(len(Vij_obs_s)), Vij_rad.mean_v, linestyle="--", color="slateblue", label="Radiation: SSI="+str(round(SSI_v_rad, 2)))
ax[0,0].fill_between(np.arange(len(Vij_obs_s)), Vij_rad.h5_v, Vij_rad.h95_v, color="slateblue", alpha=0.2)
ax[0,0].legend()
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("Average Daily Trips")
ax[0,0].set_ylabel("Estmiate")
ax[0,0].set_xlabel("Observed")

SSI_v_vis2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_vis_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_vis_m[:85]))
SSI_v_rad2 = 2*np.sum(np.minimum(Vij_obs_s[:85], Vij_rad_m[:85]))/(np.sum(Vij_obs_s[:87]) + np.sum(Vij_rad_m[:85]))
ax[0,1].plot(np.arange(len(Vij_obs_s))[:85], Vij_obs_s[:85], color='k', linestyle=":", label="Observed")
ax[0,1].plot(np.arange(len(Vij_obs_s))[:85], Vij_vis.mean_v[:85], color="g", label="Visitation: SSI="+str(round(SSI_v_vis2, 2)))
ax[0,1].fill_between(np.arange(len(Vij_obs_s))[:85], Vij_vis.h5_v[:85], Vij_vis.h95_v[:85], color="g", alpha=0.2)
ax[0,1].plot(np.arange(len(Vij_obs_s))[:85], Vij_rad.mean_v[:85], linestyle="--", color="slateblue", label="Radiation: SSI="+str(round(SSI_v_rad2, 2)))
ax[0,1].fill_between(np.arange(len(Vij_obs_s))[:85], Vij_rad.h5_v[:85], Vij_rad.h95_v[:85], color="slateblue", alpha=0.2)
ax[0,1].legend()
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("Average Daily Trips")
ax[0,1].set_ylabel("Estmiate")
ax[0,1].set_xlabel("Observed")

ax[1,0].plot(np.arange(len(Qij_obs_s)), Qij_obs_s, color='k', linestyle=":", label="Observed")
ax[1,0].plot(np.arange(len(Qij_obs_s)), Qij_vis.mean_q, color="g", label="Visitation: SSI="+str(round(SSI_q_vis, 2)))
ax[1,0].fill_between(np.arange(len(Qij_obs_s)), Qij_vis.h5_q, Qij_vis.h95_q, color="g", alpha=0.2)
ax[1,0].plot(np.arange(len(Qij_obs_s)), Qij_rad.mean_q, linestyle="--", color="slateblue", label="Radiation: SSI="+str(round(SSI_q_rad, 2)))
ax[1,0].fill_between(np.arange(len(Qij_obs_s)), Qij_rad.h5_q, Qij_rad.h95_q, color="slateblue", alpha=0.2)
ax[1,0].legend()
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("Average Daily Travellers")
ax[1,0].set_ylabel("Estmiate")
ax[1,0].set_xlabel("Observed")

SSI_q_rad2 = 2*np.sum(np.minimum(Qij_obs_s[:85], Qij_rad_m[:85]))/(np.sum(Qij_obs_s[:85]) + np.sum(Qij_rad_m[:85]))
SSI_q_vis2 = 2*np.sum(np.minimum(Qij_obs_s[:85], Qij_vis_m[:85]))/(np.sum(Qij_obs_s[:85]) + np.sum(Qij_vis_m[:85]))
ax[1,1].plot(np.arange(len(Qij_obs_s))[:85], Qij_obs_s[:85], color='k', linestyle=":", label="Observed")
ax[1,1].plot(np.arange(len(Qij_obs_s))[:85], Qij_vis.mean_q[:85], color="g", label="Visitation: SSI="+str(round(SSI_q_vis2, 2)))
ax[1,1].fill_between(np.arange(len(Qij_obs_s))[:85], Qij_vis.h5_q[:85], Qij_vis.h95_q[:85], color="g", alpha=0.2)
ax[1,1].plot(np.arange(len(Qij_obs_s))[:85], Qij_rad.mean_q[:85], linestyle="--", color="slateblue", label="Radiation: SSI="+str(round(SSI_q_rad2, 2)))
ax[1,1].fill_between(np.arange(len(Qij_obs_s))[:85], Qij_rad.h5_q[:85], Qij_rad.h95_q[:85], color="slateblue", alpha=0.2)
ax[1,1].legend()
ax[1,1].grid(alpha=0.2)
ax[1,1].set_title("Average Daily Travellers")
ax[1,1].set_ylabel("Estmiate")
ax[1,1].set_xlabel("Observed")

plt.suptitle("Visitation vs radvity: Trips and Travellers to Central Madrid", fontsize=18)
plt.tight_layout()
plt.savefig("vistation_radiation_comparison.png", dpi=300)
plt.show()
plt.close()



################### Model Comparison #################
models = {'Visitation':idata_v_y, 'Radiation':idata_r_y, 'Gravity':idata_g_y}

waic = az.compare(models, ic="waic", scale="log")
az.plot_compare(waic, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.title("Trips WAIC Model Comparison", size=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('trips_model_comp_waic.png', dpi=300)
plt.close()
waic_df = pd.DataFrame(waic)
waic_df.to_csv("trips_model_comp_waic.csv")


loo = az.compare(models, ic="loo", scale="log")
az.plot_compare(loo, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.title("Trips LOO Model Comparison", size=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('trips_model_comp_loo.png', dpi=300)
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("trips_model_comp_loo.csv")


models = {'Visitation':idata_v_w, 'Radiation':idata_r_w, 'Gravity':idata_g_w}

waic = az.compare(models, ic="waic", scale="log")
az.plot_compare(waic, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.title("Travellers WAIC Model Comparison", size=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('travellers_model_comp_waic.png', dpi=300)
plt.close()
waic_df = pd.DataFrame(waic)
waic_df.to_csv("travellers_model_comp_waic.csv")


loo = az.compare(models, ic="loo", scale="log")
az.plot_compare(loo, plot_kwargs={'color_insample_dev':'crimson', 'color_dse':'steelblue'})
plt.title("Travellers LOO Model Comparison", size=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('travellers_model_comp_loo.png', dpi=300)
plt.close()
loo_df = pd.DataFrame(loo)
loo_df.to_csv("travellers_model_comp_loo.csv")
