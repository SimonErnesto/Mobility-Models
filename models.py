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

mad = gdf[gdf.ID.str.startswith("28")] #Madrid municipios start with 28

# df = pd.read_csv("data_madrid_jan.csv")
# df = df.replace(np.nan,0)
# data = df[df.actividad_origen=="casa"]

# data['visitors'] = np.repeat(1, len(data))

# data = data.groupby(['origen', 'destino'], as_index=False).sum()

# data['ID'] = data.destino
# mad_dest = mad[['ID', 'centroid']]
# mad_dest.columns = ['ID', 'centroid_dest']
# data = pd.merge(data,mad_dest)

# data['ID'] = data.origen
# data = gpd.GeoDataFrame(pd.merge(data,mad))

# dist = data.geometry.distance(data.centroid_dest)

# data['distance'] = dist

# data['dist_km'] = dist/1000

# data.to_csv("data_jan_ave.csv", index=False)

data = pd.read_csv("data_jan_ave.csv")

pob = pd.read_csv("./data/poblacion.csv", sep="|")
pob = pob[pob.municipio.isin(data.ID.unique())]
pob = pob.drop_duplicates('municipio') 
pob['ID'] = pob.municipio

#################### Observed Data ##############################
### Plot Survey data observed
Tdf_obs = data.groupby(['origen', 'destino']).mean()
Tdf_obs = pd.pivot_table(Tdf_obs, index='origen', columns='destino', values='visitors')
Ndf = Tdf_obs[Tdf_obs.index=="28092"].T
Ndf['ID'] = Ndf.index
Ndf = Ndf.replace(np.nan,0)
madg = pd.merge(mad,Ndf)
madg = madg.sort_values(by="28092", ascending=False)
madg.reset_index(inplace=True, drop=True)

max_val = int(max(madg["28092"]))
mid_val = int(np.median(madg["28092"]))
min_val = int(min(madg["28092"]))

colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

ch_map = mad.plot(figsize=(10,20), color="tan", alpha=0.2, edgecolor="k")
for c in range(len(madg)):
    #ch_map.set_facecolor("grey")
    muni_dest = madg.ID[c]
    orig = "28092" 
    dest_coord = mad[mad.ID==muni_dest].coords.values[0]
    orig_coord = mad[mad.ID==orig].coords.values[0]
    x = (dest_coord[0], orig_coord[0])
    y = (dest_coord[1], orig_coord[1])
    ch_map.plot(x,y, color=colors[c])  
cbar = plt.colorbar(sm,fraction=0.005, pad=0.001)
cbar.set_ticklabels([min_val,mid_val,max_val])
plt.title("Observed Data Jan 2022: Travels from Móstoles", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_madrid_municipio_observed.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

obs = Tdf_obs.values
obs[np.isnan(obs)] = 0

###############################################################################
####################### Visitation Law Model ##################################

#create distance matrix for radiuses R
xy = np.array([np.array(mad.centroid.values[i].xy) for i in tqdm(range(len(mad)))]).T[0]
D = distance_matrix(xy.T, xy.T) + 0.000000001 #distance matrix in meters

R = D/1000  #radiuses in km

A = 2*np.pi*R

N = pd.pivot_table(data, index='origen', columns='destino', values='visitors').values #Number of visitors
N[np.isnan(N)] = 0

rho = N/A

F = pd.pivot_table(data, index='origen', columns='destino', values='viajes').values #frequency of visitation sum
F[np.isnan(F)] = 0.000000001

fmin = F.mean()/N.mean()/31 #once per day
fmax = F.mean()/N.mean() #max daily per person

mu = rho*(R*1)**2
Vij = (mu*A)/(np.log(fmax/fmin)*R**2) * 31

##for some reason this approx does not work
# F =  pd.pivot_table(data, index='origen', columns='destino', values='viajes')
# N =  pd.pivot_table(data, index='origen', columns='destino', values='visitors')
# fmax = (F/N)
#fmax = F.T['28092'].values/N.T['28092'].values
# rhoj = pob.poblacion.values/(mad.geometry.area.values/1e6)
# rj = (mad.geometry.area.values/1e6)/(2*np.pi)
# muj = rhoj*(rj**2)*fmax
# Ai = mad[mad.ID=="28092"].geometry.area.values/1e6 #area origin
# rij = pd.DataFrame(R, index=data.origen.unique(), columns=data.destino.unique())
# rij = rij.T["28092"].values
# Vij = (muj*Ai)/(np.log(fmax/fmin)*rij**2)


### Plot Visitation law for Mostoles
# Ndf = pd.DataFrame({"28092":Vij, "ID":data.ID.unique()})
Tdf = pd.DataFrame(Vij, index=data.origen.unique(), columns=data.destino.unique())
Ndf = Tdf[Tdf.index=="28092"].T
Ndf['ID'] = Ndf.index
Ndf = Ndf.replace(np.nan,0)
madg = pd.merge(mad,Ndf)
madg = madg.sort_values(by="28092", ascending=False)
madg.reset_index(inplace=True, drop=True)

max_val = int(max(madg["28092"]))
mid_val = int(np.median(madg["28092"]))
min_val = int(min(madg["28092"]))

colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

SSI_visit = 2*np.sum(np.minimum(Vij,obs))/(np.sum(Vij) + np.sum(obs))

ch_map = mad.plot(figsize=(10,20), color="tan", alpha=0.2, edgecolor="k")
for c in range(len(madg)):
    #ch_map.set_facecolor("grey")
    muni_dest = madg.ID[c]
    orig = "28092" 
    dest_coord = mad[mad.ID==muni_dest].coords.values[0]
    orig_coord = mad[mad.ID==orig].coords.values[0]
    x = (dest_coord[0], orig_coord[0])
    y = (dest_coord[1], orig_coord[1])
    ch_map.plot(x,y, color=colors[c])  
cbar = plt.colorbar(sm,fraction=0.005, pad=0.001)
cbar.set_ticklabels([min_val,mid_val,max_val])
plt.text(0.1, 0.1,"SSI: "+str(round(SSI_visit, 2)), transform=ch_map.transAxes, fontsize=18)
plt.title("Visitation Law Estimate: Average Daily Travellers from Móstoles Jan 2022", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_madrid_municipio_visitation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()



###############################################################################
######################## Radiation Model ######################################
pob.reset_index(inplace=True, drop=True)
pobs = []
for i in tqdm(range(len(pob))):
    ori = pob.municipio[i]
    des = pob.municipio.values
    p = pob.poblacion.values
    ps = pd.DataFrame({'origen':np.repeat(ori, len(pob)), 'destino':des, 'poblacion':p})
    pobs.append(ps)
pobs = pd.concat(pobs)
P = pd.pivot_table(pobs, index='origen', columns='destino', values='poblacion')
P = P.values

## (create surrounding population matrix Sij) THIS ONE DOES NOT WORK!
def calc_pops(R,P):
    Sij0 = []
    for i in tqdm(range(R.shape[0])):
        r = R[i,:]
        s = [R[i,:]<=r[j] for j in range(R.shape[1])] #exclude places outside radius
        # calculate populations sum within radious excluding origin and destination pops 
        p = [P[i,:][s[j]].sum() -P[i,j] for j in range(R.shape[1])]
        Sij0.append(p)
    return np.array(Sij0)    
Sij = calc_pops(R,P)


N = pob.poblacion.values.sum()
Nc = data.visitors.sum() #total number of commuters over 1 month period
Mi = pob.poblacion.values #populations origin
Nj = pob.poblacion.values #populations destination
Ti = Mi*(Nc/N) #estimated flux from origin

#Radiation Model
p_num = Mi*Nj
p_den = (Mi + Sij)*(Mi + Nj + Sij)
p = p_num/p_den
Tmean = Ti*p
Tsd = np.sqrt(Ti*p*(1-p))



### Plot Radiation Model's Estimates for Mostoles
Tdf = pd.DataFrame(Tmean, index=mad.ID.unique(), columns=mad.ID.unique())
Ndf = Tdf[Tdf.index=="28092"].T
Ndf['ID'] = Ndf.index
Ndf = Ndf.replace(np.nan,0)
madg = pd.merge(mad,Ndf)
madg = madg.sort_values(by="28092", ascending=False)
madg.reset_index(inplace=True, drop=True)

max_val = int(max(madg["28092"]))
mid_val = int(np.median(madg["28092"]))
min_val = int(min(madg["28092"]))

colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

SSI_rad = 2*np.sum(np.minimum(Tmean,obs))/(np.sum(Tmean) + np.sum(obs))

ch_map = mad.plot(figsize=(10,20), color="tan", alpha=0.2, edgecolor="k")
for c in range(len(madg)):
    #ch_map.set_facecolor("grey")
    muni_dest = madg.ID[c]
    orig = "28092" 
    dest_coord = mad[mad.ID==muni_dest].coords.values[0]
    orig_coord = mad[mad.ID==orig].coords.values[0]
    x = (dest_coord[0], orig_coord[0])
    y = (dest_coord[1], orig_coord[1])
    ch_map.plot(x,y, color=colors[c])  
cbar = plt.colorbar(sm,fraction=0.005, pad=0.001)
cbar.set_ticklabels([min_val,mid_val,max_val])
plt.text(0.1, 0.1,"SSI: "+str(round(SSI_rad, 2)), transform=ch_map.transAxes, fontsize=18)
plt.title("Radiation Model Estimate: Average Daily Travellers from Móstoles Jan 2022", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_madrid_municipio_radiation.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

###############################################################################
########################### Gavity Model ######################################

Mi = pob.poblacion.values #populations origin
Nj = pob.poblacion.values #populations destination
muni_n = len(data.origen.unique())

##single origin version
dj = pd.DataFrame(R, index=mad.ID.unique(), columns=mad.ID.unique())
dj = dj.T['28092'].values
T = Tdf_obs.T['28092'].values
with pm.Model() as mod:
    t_s = pm.Exponential("t_s", 3)
    theta = pm.HalfNormal("theta", t_s, shape=muni_n)
    o_s = pm.Exponential("o_s", 3)
    omega = pm.HalfNormal("omega", o_s, shape=muni_n)
    g_s = pm.Exponential("g_s", 3)
    gamma = pm.HalfNormal("gamma", g_s, shape=muni_n)
    lam_den = (Nj**omega)*(dj**-gamma)
    lam_num = at.sum(lam_den)
    lam = pm.Deterministic("lam", theta*Mi*(lam_den/lam_num))
    alpha = pm.Exponential("alpha", 50)
    m = pm.NegativeBinomial("m", mu=lam, alpha=alpha, observed=T)

dag = pm.model_to_graphviz(mod)
dag.render("gravity_model_dag", format="png")
dag

with mod:
    idata = pm.sample(2000, chains=4, cores=12)#, target_accept=0.9)

pos = idata.stack(sample = ['chain', 'draw']).posterior
pos_lam = pos['lam'].values.mean(axis=1)
Ndf =  pd.DataFrame({"28092":pos_lam})
Ndf['ID'] = data.destino.unique()

# with pm.Model() as mod:
#     t_s = pm.Exponential("t_s", 3)
#     theta = pm.HalfNormal("theta", t_s, shape=muni_n)
#     o_s = pm.Exponential("o_s", 3)
#     omega = pm.HalfNormal("omega", o_s, shape=muni_n)
#     g_s = pm.Exponential("g_s", 3)
#     gamma = pm.HalfNormal("gamma", g_s, shape=muni_n)
#     lam_den = (Nj**omega)*(R**-gamma)
#     lam_num = at.sum(lam_den, axis=1)
#     lam = pm.Deterministic("lam", theta*Mi*(lam_den/lam_num))
#     alpha = pm.Exponential("alpha", 10)
#     m = pm.NegativeBinomial("m", mu=lam, alpha=alpha, observed=Tdf_obs.values)
    
# dag = pm.model_to_graphviz(mod)
# dag.render("gravity_model_dag", format="png")
# dag

# with mod:
#     idata = pm.sample(2000, chains=4, cores=12, target_accept=0.99)
    
# pos = idata.stack(sample = ['chain', 'draw']).posterior

# pos_lam = pos['lam'].values.mean(axis=2)


### Plot Gravity Model's Estimates for New York County
# Tdf =  pd.DataFrame(pos_lam, index=data.origen.unique(), columns=data.destino.unique())
# Ndf = Tdf[Tdf.index=="28092"].T
# Ndf['ID'] = Ndf.index
Ndf = Ndf.replace(np.nan,0)
madg = pd.merge(mad,Ndf)
madg = madg.sort_values(by="28092", ascending=False)
madg.reset_index(inplace=True, drop=True)

max_val = int(max(madg["28092"]))
mid_val = int(np.median(madg["28092"]))
min_val = int(min(madg["28092"]))

colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

SSI_grav = 2*np.sum(np.minimum(pos_lam,obs))/(np.sum(pos_lam) + np.sum(obs))

ch_map = mad.plot(figsize=(10,20), color="tan", alpha=0.2, edgecolor="k")
for c in range(len(madg)):
    #ch_map.set_facecolor("grey")
    muni_dest = madg.ID[c]
    orig = "28092" 
    dest_coord = mad[mad.ID==muni_dest].coords.values[0]
    orig_coord = mad[mad.ID==orig].coords.values[0]
    x = (dest_coord[0], orig_coord[0])
    y = (dest_coord[1], orig_coord[1])
    ch_map.plot(x,y, color=colors[c])  
cbar = plt.colorbar(sm,fraction=0.005, pad=0.001)
cbar.set_ticklabels([min_val,mid_val,max_val])
plt.text(0.1, 0.1,"SSI: "+str(round(SSI_grav, 2)), transform=ch_map.transAxes, fontsize=18)
plt.title("Gravity Model Estimate: Average Daily Travellers from Móstoles Jan 2022", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("plot_madrid_municipio_gravity.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

#save summary of mcmc posteriors
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("gravity_model_hierarchical_summary.csv")