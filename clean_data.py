# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import tarfile
from tqdm import tqdm
import glob

os.chdir(os.getcwd())


gdf = gpd.read_file("./data/zonificacion_municipios.shp")
gdf = gdf.to_crs('+proj=cea')

gdf.reset_index(inplace=True, drop=True)
gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
gdf['coords'] = [coords[0] for coords in gdf['coords']]


mad = gdf[gdf.ID.str.contains("28")]

mad = mad[mad.ID.str.startswith("28")]

#extract data from Madrid only
tar_files = glob.glob("./data/*.tar")
dfs = []
for tar in tar_files:
    files = []
    tars = tarfile.open(tar)
    for member in tqdm(tars.getmembers()):
        f = tars.extractfile(member)
        f = pd.read_csv(f, compression='gzip', sep='|')
        f = f[f.origen.str.contains("28")]
        f = f[f.origen.str.startswith("28")]
        f = f[f.destino.str.contains("28")]
        f = f[f.destino.str.startswith("28")]
        files.append(f)
        df = pd.concat(files)
    df['visitors'] = np.repeat(1, len(df))
    dfs.append(df.groupby(["origen", "destino"], as_index=False).mean())

df = pd.concat(dfs)
df = df.groupby(["origen", "destino"], as_index=False).mean()
df.to_csv("data_madrid_2022_ave.csv", index=False)


df = pd.concat(dfs)
df.to_csv("data_madrid_2022_monthly_ave.csv", index=False)

# f = files[0]    


# ### Plot Survey data observed
# Tdf = f.groupby(['origen', 'destino']).mean()
# Tdf = pd.pivot_table(Tdf, index='origen', columns='destino', values='viajes')
# Ndf = Tdf[Tdf.index=="28092"].T
# Ndf['ID'] = Ndf.index
# Ndf = Ndf.replace(np.nan,0)
# madg = pd.merge(mad,Ndf)
# madg = madg.sort_values(by="28092", ascending=False)
# madg.reset_index(inplace=True, drop=True)

# max_val = int(max(madg["28092"]))
# mid_val = int(np.median(madg["28092"]))
# min_val = int(min(madg["28092"]))

# colors = np.array([mpl.cm.get_cmap('gist_heat')(x/len(madg)) for x in range(len(madg))])
# sm = plt.cm.ScalarMappable(cmap=mpl.cm.get_cmap('gist_heat'))

# ch_map = mad.plot(figsize=(10,20), color="tan", alpha=0.2, edgecolor="k")
# for c in range(len(madg)):
#     #ch_map.set_facecolor("grey")
#     muni_dest = madg.ID[c]
#     orig = "28092" 
#     dest_coord = mad[mad.ID==muni_dest].coords.values[0]
#     orig_coord = mad[mad.ID==orig].coords.values[0]
#     x = (dest_coord[0], orig_coord[0])
#     y = (dest_coord[1], orig_coord[1])
#     ch_map.plot(x,y, color=colors[c])  
# cbar = plt.colorbar(sm,fraction=0.005, pad=0.001)
# cbar.set_ticklabels([min_val,mid_val,max_val])
# plt.title("Observed Data 01/01/2022 : Travels from Móstoles")
# plt.axis('off')
# plt.tight_layout()
# plt.savefig("plot_madrid_municipio_test.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()
# plt.close()
