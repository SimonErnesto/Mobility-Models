# -*- coding: utf-8 -*-
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
from scipy.stats import binned_statistic

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
data = data[data.destino==madri] #data.groupby(["origen", "destino"], as_index=False).mean()
data['ID'] = data.origen

pob = pd.read_csv("./data/poblacion.csv", sep="|")
pob['ID'] = pob.municipio
pob = pob[pob.municipio.isin(data.ID.unique())]
pob = pob.drop_duplicates('municipio') 

data = pd.merge(data, pob)
madri_idx = data[data.origen==madri].index[0]


###############################################################################
####################### Visitation Law Model ##################################

#create distance matrix for radiuses R
xy = np.array([np.array(mad.centroid.values[i].xy) for i in tqdm(range(len(mad)))]).T[0]
D = distance_matrix(xy.T, xy.T) + 0.000000001 #distance matrix in meters
R = D/1e3  #radiuses in km

fmin = 1/31 #data.viajes/data.visitors/31 #minimum frequency
fmax = 1 #data.viajes/data.visitors
Nj = pob[pob.municipio==madri].poblacion.values #population destination
Njv = data.visitors.values #visitors destination from origins
Ai = mad.geometry.area.values/1e6 #area origin
Aj = Ai[madri_idx] #area destination
rj = Aj/(2*np.pi)#radius destination
rhoj = Nj/Aj
rij = R.T[madri_idx] #radiuses from origins to central Madrid
rij[madri_idx] = rj 
muj = rhoj*(rj*1)**2

Vij = (muj*Ai)/(np.log(fmax/fmin)*rij**2) * 31 #compute monthly average number of trips

Qij = Ai*(31**-1)*muj/(rj**2) #compute number of unique visitors

Vij_vis = np.sort(Vij)[:118]
Qij_vis = np.sort(Qij)

Vij_obs = np.sort(data.viajes.values)[:118]
Qij_obs = np.sort(data.visitors.values)

SSI_v_vis = 2*np.sum(np.minimum(Vij_obs, Vij_vis))/(np.sum(Vij_obs) + np.sum(Vij_vis))
SSI_q_vis = 2*np.sum(np.minimum(Qij_obs, Qij_vis))/(np.sum(Qij_obs) + np.sum(Qij_vis))


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16,8))
ax[0,0].plot(Vij_obs, Vij_obs, color='k', linestyle=":", label="Observed")
ax[0,0].scatter(Vij_obs, Vij_vis, marker="o", facecolor="w", color="g", label="Daily Trips: SSI="+str(round(SSI_v_vis, 2)))
ax[0,0].legend()
ax[0,0].grid(alpha=0.2)
ax[0,0].set_title("Average Daily Trips")
ax[0,0].set_ylabel("Estmiate")
ax[0,0].set_xlabel("Observed")
ax[0,1].plot(Vij_obs[:85], Vij_obs[:85], color='k', linestyle=":", label="Observed")
SSI_v_vis = 2*np.sum(np.minimum(Vij_obs[:85], Vij_vis[:85]))/(np.sum(Vij_obs[:85]) + np.sum(Vij_vis[:85]))
ax[0,1].scatter(Vij_obs[:85], Vij_vis[:85], marker="o", facecolor="w", color="g", label="Daily Trips: SSI="+str(round(SSI_v_vis, 2)))
ax[0,1].legend()
ax[0,1].grid(alpha=0.2)
ax[0,1].set_title("Average Daily Trips < 50000")
ax[0,1].set_ylabel("Estmiate")
ax[0,1].set_xlabel("Observed")
ax[1,0].plot(Qij_obs, Qij_obs, color='k', linestyle=":", label="Observed")
ax[1,0].scatter(Qij_obs, Qij_vis, marker="o", facecolor="w", color="g", label="Daily Visitors: SSI="+str(round(SSI_q_vis, 2)))
ax[1,0].legend()
ax[1,0].grid(alpha=0.2)
ax[1,0].set_title("Average Daily Visitors")
ax[1,0].set_ylabel("Estmiate")
ax[1,0].set_xlabel("Observed")
ax[1,1].plot(Qij_obs[:87], Qij_obs[:87], color='k', linestyle=":", label="Observed")
SSI_q_vis = 2*np.sum(np.minimum(Qij_obs[:87], Qij_vis[:87]))/(np.sum(Qij_obs[:87]) + np.sum(Qij_vis[:87]))
ax[1,1].scatter(Qij_obs[:87], Qij_vis[:87], marker="o", facecolor="w", color="g", label="Daily Visitors: SSI="+str(round(SSI_q_vis, 2)))
ax[1,1].legend()
ax[1,1].grid(alpha=0.2)
ax[1,1].set_title("Average Daily Visitors < 15000")
ax[1,1].set_ylabel("Estmiate")
ax[1,1].set_xlabel("Observed")
plt.suptitle("Visitation Law Approximation: Trips and Travellers to Central Madrid", fontsize=18)
plt.tight_layout()
plt.savefig("visitation_law_estimates.png", dpi=300)
plt.show()
plt.close()
