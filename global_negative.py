# %load start_to_finish.py
import time
import deepmars2.post_processing_net.model as ppn
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
from subprocess import Popen
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsClassifier
from joblib import Parallel, delayed
import tifffile
from tqdm import tqdm
from keras.models import load_model
from multiprocessing import Process, Queue
import deepmars2.data.data as data
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import deepmars2.config as cfg
import cratertools.metric as metric
import sys
import rasterio

min_box_size = 1
max_box_size = 1.587
n_box_sizes = 3
min_lat = -90
max_lat = 90
#min_box_size = 5
#max_box_size = 15
#n_box_sizes = 3
#min_lat = -40
#max_lat = 40
min_long = -180
max_long = 180
cols = ['Long', 'Lat', 'Diameter (km)']
def load_DEM_IR():
    #DEM = tifffile.imread("./source_data/kermadec_volcanoes_test/Kermaded_DEM_test.tif")
#    DEM_src = tifffile.imread("./source_data/gebco/gebco_2023_clipped_to_seamounts_proj.tif")
    DEM_src = rasterio.open("./source_data/gebco/gebco_2023_clipped_to_seamounts_proj.tif")
    DEM = DEM_src.read(1)
    #global min_lat, max_lat, min_long, max_long, min_box_size, max_box_size
    #min_lat = 0
    #max_lat = DEM.shape[0]
    #min_long = 0
    #max_long = DEM.shape[1]
    #min_box_size = min(DEM.shape)//10
    #max_box_size = min(DEM.shape)//2
    print("BOX SIZE", min_box_size, max_box_size)
    return DEM, DEM_src, None, None


def UNET_predictions():
    # Can be re-run if not all iterations are finished successfully
    DEM, DEM_src, _a, _b = load_DEM_IR()
    STEP=1000
    #print(min_lat, max_lat, min_long, max_long, min_box_size, max_box_size)
    box_sizes = np.exp(np.linspace(np.log(min_box_size), np.log(max_box_size), n_box_sizes))#.astype(int)
    print('Box sizes: ', box_sizes.round(1))
    print(box_sizes, min_lat, max_lat,min_long, max_long)
    sys_pass = data.systematic_pass(box_sizes, min_lat, max_lat, 
                                    min_long, max_long,
                                   project=True)
    def test_box(lat_0, lon_0, box_size, img):
        #short circuit for missing data
        ny,nx = img.shape
        cx,cy = lon_0, lat_0
        
        approx_lat_pix = (ny/180)*(cy+90)
        approx_lon_pix = (nx/360)*(cx+180)
        bs_pix = box_size * nx/360
    
        sx = slice(int(approx_lon_pix-bs_pix), int(approx_lon_pix+bs_pix))
        sy = slice(int(approx_lat_pix-bs_pix), int(approx_lat_pix+bs_pix))
        if sx.start < 0 or sx.stop > nx or sy.start < 0 or sy.stop > ny:
            return False
        
        d = img[sy,sx]
        if d.max()==d.min():
            return False
        return True
    #
    print(len(sys_pass))
#    import pickle
#    if os.path.exists("sys_pass.pkl"):
#        sys_pass = pickle.load(open("sys_pass.pkl",'rb'))
#    else:
    if True:
        sys_pass = [s for s in tqdm(sys_pass) if test_box(s[0],s[1], s[2],DEM)]
#        pickle.dump(sys_pass, open("sys_pass.pkl","wb"))
        
    print(len(sys_pass))
    import sys
    n_files = len(sys_pass) // STEP + 1
    empty_craters = pd.DataFrame(columns=['Long', 'Lat', 'Diameter (km)'])
    # Cell can be re-run if there are missing files

    for i in range(n_files):
        start_index = i * STEP
        do_DEM = not os.path.isfile('./data/negative_predictions/DEM/sys_cal_craterdist_{:05d}.npy'.format(start_index))
        # Generate images
        
        if do_DEM:
            print('Making dataset for file {:05d}'.format(start_index), flush=True)
            data.gen_dataset(DEM, DEM_src, None, None, empty_craters, 'sys_cal', start_index, 'systematic',
                         sys_pass=sys_pass, in_notebook=False,
                         min_lat=min_lat, max_lat=max_lat, min_long=min_long, max_long=max_long,
                             project=True, amount=STEP)
   
            if do_DEM:
                print('Making Predictions (IR and DEM) {}'.format(start_index), flush=True)
                Popen(["./cnn_ir_dem_negative.bash",str(start_index)])


UNET_predictions()
