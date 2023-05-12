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

pob = pd.read_csv("./data/poblacion.csv", sep="|")
pob = pob[pob.municipio.isin(data.ID.unique())]
pob = pob.drop_duplicates('municipio') 
pob['ID'] = pob.municipio

#################### Observed Data ##############################
### Plot Survey data observed
Tdf_obs = data.groupby(['origen', 'destino']).mean()
Tdf_obs = pd.pivot_table(Tdf_obs, index='origen', columns='destino', values='visitors')
madri_idx = Tdf_obs.columns.get_loc(madri)
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
plt.savefig("plot_madrid_municipio_observed.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()


###############################################################################
####################### Visitation Law Model ##################################

#create distance matrix for radiuses R
xy = np.array([np.array(mad.centroid.values[i].xy) for i in tqdm(range(len(mad)))]).T[0]
D = distance_matrix(xy.T, xy.T) + 0.000000001 #distance matrix in meters

R = D/1e4  #radiuses in km

Ai = mad.geometry.area.values/1e8 #area origin
rhoj = pob.poblacion.values/Ai[madri_idx]
rhoj = rhoj[madri_idx] #flow to central Madrid
rj = Ai/(2*np.pi)
rj = rj[madri_idx]
rij = R.T[madri_idx] #radiuses from origins to central Madrid
rij[madri_idx] = rj 

F = pd.pivot_table(data, index='origen', columns='destino', values='viajes').values #frequency of visitation sum
F[np.isnan(F)] = 0.000000001

T = Tdf_obs.values

with pm.Model() as mod:
    mu = pm.Uniform("mu", 0, 10)
    fmin = pm.Uniform("fmin", 0, 1)
    fmax = pm.Uniform("fmax", 0, 10)
    v = pm.Deterministic("v", (mu*Ai)/(np.log(fmax/fmin)*rij**2))
    alpha = pm.HalfNormal("alpha", 10)
    y = pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=T)

with mod:
    idata = pm.sample(1000)

pos = idata.stack(sample = ['chain', 'draw']).posterior

Vij = pos['v'].values.mean(axis=1)


### Plot Visitation law for Mostoles
Ndf = pd.DataFrame({madri:Vij, "ID":mad.ID.unique()})
#Tdf = pd.DataFrame(Vij, index=data.origen.unique(), columns=data.destino.unique())
#Ndf = Tdf #Tdf[Tdf.index=="28092"].T
#Ndf['ID'] = Ndf.index
Ndf = Ndf.replace(np.nan,0)
madg = pd.merge(mad,Ndf)
madg = madg.sort_values(by=madri, ascending=False)
madg.reset_index(inplace=True, drop=True)

max_val = int(max(madg[madri]))
mid_val = int(np.median(madg[madri]))
min_val = int(min(madg[madri]))

colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

obs = Tdf_obs.values

SSI_vis = 2*np.sum(np.minimum(Vij, obs[madri_idx]))/(np.sum(Vij) + np.sum(obs[madri_idx]))


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
plt.text(0.1, 0.1,"SSI: "+str(round(SSI_vis, 2)), transform=ch_map.transAxes, fontsize=18)
plt.title("Visitation Law Estimate: Average Daily Travellers to Central Madrid Jan 2022", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_madrid_municipio_visitation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
