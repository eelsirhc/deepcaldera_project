import os
os.chdir("/mnt/export/lee/1-Projects/deepcaldera")
import numpy as np
import pandas as pd
import deepmars2.data.data as data
import deepmars2.config as cfg
import tifffile
import matplotlib.pyplot as plt
import cratertools.metric as metric
from tqdm import tqdm_notebook
from sklearn.ensemble import GradientBoostingClassifier
import cratertools.metric as metric
import deepmars2.post_processing_net.model as ppn
from keras.models import load_model
import os
import click




n_files = 167
craters_np = np.empty([0,4])
cols = ['Long', 'Lat', 'Diameter (km)', "file"]
for i in range(0,n_files+1):
    crater_file_name = './data/low/negative_predictions/DEM/sys_cal_craterdist_{:05d}.npy'.format(i * 1000)
    newdata = np.load(crater_file_name)
    #print(len(newdata))
    if newdata.shape[0]>0:
        newdata = np.hstack([newdata,(np.ones(newdata.shape[0], dtype=int)*i)[:,None]])
        #newdata = np.hstack([newdata, )
        craters_np = np.vstack([craters_np, newdata])
#-----
n_files = 175
craters_np = np.empty([0,4])
cols = ['Long', 'Lat', 'Diameter (km)', "file"]
for i in range(0,n_files+1):
    crater_file_name = './data/negative_predictions/DEM/sys_cal_craterdist_{:05d}.npy'.format(i * 1000)
    newdata = np.load(crater_file_name)
    #print(len(newdata))
    if newdata.shape[0]>0:
        newdata = np.hstack([newdata,(np.ones(newdata.shape[0], dtype=int)*i)[:,None]])
        #newdata = np.hstack([newdata, )
        craters_np = np.vstack([craters_np, newdata])
#----
        
craters_np[:,2] *= 2 # convert radii to diameters
my_craters_DEM = pd.DataFrame(craters_np, columns=cols)

# Combining lists

my_craters_DEM['DEM'] = 1

my_craters_combined = my_craters_DEM
my_craters_combined = my_craters_combined[["Long","Lat","Diameter (km)", "DEM", "file"]]

# Crater filtering
my_craters_combined_filtered = metric.rep_filter_unique_craters(my_craters_combined, *cols[:3])[0]

# Crater averaging

things = metric.kn_match_craters(my_craters_combined_filtered, my_craters_combined, *cols[:3], max_neighbours=20)
avg_combined = [my_craters_combined.iloc[things[2][loc][:things[3][loc]]].mean() for loc in things[2].keys()]
avg_combined = pd.DataFrame(avg_combined)
avg_combined['duplicates'] = things[0]['l']

# Extra duplicate removal

filtered = metric.rep_filter_unique_craters(avg_combined, *cols[:3])[0]
avg_combined_filtered = pd.merge(avg_combined, filtered, how='left', on=cols[:3], indicator='ind')
avg_combined_filtered = avg_combined_filtered[avg_combined_filtered['ind'] == 'both'].copy()
avg_combined_filtered.drop('ind', axis=1, inplace=True)

avg_combined_filtered.sort_values("duplicates")

avg_combined_filtered.to_csv("caldera_negative.csv")

@click.group()
def postprocess():
    pass

@postprocess.command()
@click.option("--resolution", default=None)
@click.option("--negative",default=False, is_flag=True)
def list(resolution, negative):
    root_dir="./"
    if resolution is not None:
        data_tld = os.path.join(root_dir, resolution)
    else:
        data_tld = root_dir

    from pathlib import Path
    if negative:
        directory=Path(data_tld)/"negative_predictions/DEM/"
    else:
        directory=Path(data_tld)/"predictions/DEM/"

    for fname in sorted(directory.glob("sys_cal_craterdist*npy")):
        print(fname)
    
        
if __name__ == "__main__":
    postprocess()

