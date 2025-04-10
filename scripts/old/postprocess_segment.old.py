import os
os.chdir("/mnt/export/lee/1-Projects/deepcaldera")
import numpy as np
import pandas as pd
import deepmars2.data.data as data
import deepmars2.config as cfg
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import os
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import fiona.transform
from rasterio.transform import from_bounds, AffineTransformer
import rasterio

def reproject_coords(src_crs, dst_crs, coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    nxs, nys = fiona.transform.transform(src_crs, dst_crs, xs, ys)
    return [[x,y] for x,y in zip(nxs, nys)]

def cross_section(row, img, src, dim=256):
    """Creates an orthographic projection from a plate caree projection.

    Paramters
    ---------
    img : numpy.ndarray
        The original image in plate carree coordinates to project from.
    dim : int, optional
        The width/height of the output image.  Only square outputs are
        supported.

    Returns
    -------
    ortho : numpy.ndarray
        The orthographic projection.
    """
    #convert from lat long to ortho
    lon_0, lat_0, diameter_km = row.Long, row.Lat, row["Diameter (km)"]
    box_size_km= 3*diameter_km
    planet_rad_km = 6371
    km_to_deg = 360/(2*np.pi*planet_rad_km * np.cos(np.deg2rad(row.Lat)))
    box_size_deg = km_to_deg * box_size_km #approx box
    box_size = box_size_deg
    
    ##1. extract the ortho image
    coords = [[lon_0, lat_0]]

    mercator = "EPSG:3395"
    latlong = "EPSG:4326"
    orthographic = dict(proj="ortho", lat_0=lat_0,lon_0=lon_0)

    #get the box limits in lat,lon to ortho
    centre = reproject_coords(latlong, orthographic, [[lon_0, lat_0]])[0]
    top = reproject_coords(latlong, orthographic, [[lon_0, lat_0+box_size/2]])[0]
    bottom = reproject_coords(latlong, orthographic, [[lon_0, lat_0-box_size/2]])[0]
    width = top[1]-bottom[1]
    
    new_left, new_bottom, new_right, new_top = centre[0]-width/2, bottom[1], centre[0]+width/2, top[1]

    #  want an output raster of width x height pixels:
    dst_width, dst_height = dim,dim
    # Create the affine transform for the destination raster:
    dst_transform = from_bounds(new_left, new_bottom, new_right, new_top, dst_width, dst_height)
    src_transform = src.transform
    src_crs = src.crs

    # Prepare an empty array for the destination data
    dst_data = np.empty((dst_height, dst_width), dtype=img.dtype)
    dst_crs = orthographic
    # # Reproject the data:
    pp = reproject(
        source=img,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest  # or another resampling method as needed
    )
    #Now make circles on the data
    tr = AffineTransformer(dst_transform)
    #samples

    #inverse the transform
    inv_transform = ~dst_transform
    
    angles = [0,45,90,135]
    lines = dict()
    npoints = None
    from rasterio import sample
    #lines
    rad = diameter_km *1e3 / 2
    x0, y0 = 0,0 #in the image coords
    def collect_points(points):
        image_points = inv_transform * points
        row,col = np.round(image_points[0]).astype(int),np.round(image_points[1]).astype(int)
        image_samples = dst_data[row,col]
        return image_points, image_samples
    
    for ang in angles:
        updown = 0
        peaked = False
        
        ath = np.deg2rad(ang)
        if npoints is None:
            npoints = 64
        radii = np.linspace(-rad*2,rad*2, npoints)
        #sample points
        points = x0+radii*np.cos(ath),y0+radii*np.sin(ath)
        line_points, line_samples = collect_points(points)
        

        
        lines[ang] = dict(x=points[0],y=points[1],r=radii,
                          ix=line_points[0], iy=line_points[1],z=line_samples,
                          variable=np.std(line_samples),
                         )
    #circles
    circles=dict()
    for circrad in [0.1,0.75,1.0,1.25]:
        Circpoints = np.linspace(0,2*np.pi,npoints)
        cc,sc =  circrad*rad*np.cos(Circpoints), circrad*rad*np.sin(Circpoints)
        c_points, c_samples = collect_points((x0+cc, y0+sc))
        circles[circrad] = dict(ix=c_points[0], iy=c_points[1],z=c_samples, rad=circrad)

    meta = dict()

    lon_ortho = dst_transform*(np.arange(dst_width), np.zeros(dst_height))
    lat_ortho = dst_transform*(np.zeros(dst_width), np.arange(dst_height))
    lon_ortho = list(zip(*lon_ortho))
    lat_ortho = list(zip(*lat_ortho))
    #print(lon_ortho), len(lat_ortho))
    _lons = reproject_coords(orthographic, latlong, lon_ortho)
    _lats = reproject_coords(orthographic, latlong, lat_ortho)
    meta["min"] = pp[0].min()
    meta["max"] = pp[0].max()
    meta["mean"] = pp[0].mean()
    meta["median"] = np.median(pp[0])
    meta["ring_mean"] = circles[1.0]["z"].mean()
    meta["centre_mean"] = circles[0.1]["z"].mean()
    meta["inner_mean"] = circles[0.75]["z"].mean()
    meta["outer_mean"] = circles[1.25]["z"].mean()

    meta["ring_std"] = circles[1.0]["z"].std()
    meta["centre_std"] = circles[0.1]["z"].std()
    meta["inner_std"] = circles[0.75]["z"].std()
    meta["outer_std"] = circles[1.25]["z"].std()

    meta["ridge"] = np.nanmean(circles[0.75]["z"])>np.nanmean(circles[1.25]["z"])
    meta["peak"] = np.nanmean(circles[0.1]["z"])>np.nanmean(circles[0.75]["z"])
    
    coords = dict(lons=[x for x,y in _lons],
                  lats=[y for x,y in _lats])
    #There HAS to be a better way
    
    #print( "meta",  meta["lons"],dst_transform*(np.arange(dst_width), np.zeros(dst_height)))
                                
    return pp[0],lines,circles,meta,coords


avg_combined_filtered = pd.read_csv("caldera_negative.csv",index_col=0)
#filter to reduce number

avg = avg_combined_filtered
th=10
avg=avg[avg.duplicates>(th-np.log2(avg["Diameter (km)"]))]
print(f"processing {len(avg)} entries")

#load the source file
from pathlib import Path
import sys
import pickle
from tqdm import tqdm
low,high = int(sys.argv[1]),int(sys.argv[2])
if low==0:
    avg.to_csv("avg.csv")
#sys.exit(0)

avg = avg.sort_values("duplicates",ascending=False)

filename = Path("/mnt/export/lee/1-Projects/deepcaldera/source_data/gebco/gebco_2023_clipped_to_seamounts_proj.tif")
src = rasterio.open(filename)
src_data= src.read(1)

      
count = 0


_avg = avg.iloc[low:high]
addon = dict()
results = Path("data/postprocessed")

for irow, row in tqdm(_avg.iterrows(),total=len(_avg)):
    fname=results/f"{irow}.pkl"
    if fname.exists():
        d = pickle.load(open(fname,'rb'))
        addon[irow]=d["meta"]
    else:
        ortho, lines,circles, meta,coords = cross_section(row, src_data, src, dim=256)
        addon[irow]=meta
        d=dict(ortho=ortho, lines=lines, circles=circles, meta=meta, coords=coords, irow=irow)
        pickle.dump(d,open(fname,'wb'))

addon_df = pd.DataFrame(addon).T
df = pd.merge(_avg,addon_df,left_index=True, right_index=True)

df.to_csv(results/f"{low:05d}_{high:05d}.csv")

    



