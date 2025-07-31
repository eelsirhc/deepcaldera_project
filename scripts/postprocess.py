import os
os.chdir("/mnt/export/lee/1-Projects/deepcaldera")
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import os
import click
import cratertools.metric as metric
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.windows import Window, transform as window_transform
from datetime import datetime
from pathlib import Path

# Define suffix
suffix = "postprocessed"

# Get current date and hour
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%Hh")
timestamp = "2025-06-19_10h"

# Create directory name
dirname = f"data/{suffix}_{timestamp}"
output_directory = Path(dirname)

output_directory.mkdir(parents=True, exist_ok=True)

@click.group()
def postprocess():
    pass

@postprocess.command()
@click.option("--resolution", default=None)
@click.option("--negative",default=False, is_flag=True)
def list(resolution, negative):
    print("Resolution: ", resolution)
    print("negative: ", negative)
    root_dir="./"
    if resolution is not None:
        data_tld = os.path.join(root_dir, resolution)
    else:
        data_tld = root_dir

    from pathlib import Path
    if negative:
        directory=Path(data_tld)/"data/negative_predictions/DEM/"
    else:
        directory=Path(data_tld)/"data/predictions/DEM/"
    print(directory)
    for fname in sorted(directory.glob("sys_cal_craterdist*npy")):
        print(fname)

def get_files(resolution, negative):
    if resolution=="all":
        resolutions=["lowres","highres"]
        files=[]
        for res in resolutions:
            files.extend(get_files(res,negative))
        return files
    
    print("Resolution: ", resolution)
    print("negative: ", negative)
    root_dir="./"
    if resolution is not None:
        data_tld = os.path.join(root_dir, resolution)
    else:
        data_tld = root_dir

    from pathlib import Path
    if negative:
        directory=Path(data_tld)/"data/negative_predictions/DEM/"
    else:
        directory=Path(data_tld)/"data/predictions/DEM/"
    print(directory)

    return sorted(directory.glob("sys_cal_craterdist*npy"))

    
@postprocess.command()
@click.option("--resolution", default=None)
@click.option("--negative",default=False, is_flag=True)
def list(resolution, negative):
    for fname in get_files(resolution,negative):
        print(fname)

@postprocess.command()
@click.option("--resolution", default="all")
@click.option("--negative",default=False, is_flag=True)
@click.option("--output", default="caldera.csv")
def combine(resolution, negative,output):
    craters_np = np.empty([0,4])
    cols = ['Long', 'Lat', 'Diameter (km)', "file"]
    for fname in get_files(resolution,negative):
        print(fname)
        newdata = np.load(fname)
        #print(len(newdata))
        if newdata.shape[0]>0:
            index=int(fname.stem.split("_")[-1])
            newdata = np.hstack([newdata,(np.ones(newdata.shape[0], dtype=int)*index)[:,None]])
            #newdata = np.hstack([newdata, )
            craters_np = np.vstack([craters_np, newdata])

    craters_np[:,2] *= 2 # convert radii to diameters
    my_craters_DEM = pd.DataFrame(craters_np, columns=cols)

    # Combining lists
    my_craters_DEM['DEM'] = 1
    my_craters_combined = my_craters_DEM
    my_craters_combined = my_craters_combined[["Long","Lat","Diameter (km)", "DEM", "file"]]

    my_craters_combined.to_csv(output_directory/f"mcc.csv")

    my_craters_combined =  my_craters_combined.dropna()

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
    avg_combined_filtered.to_csv(output_directory/output)

#----
from pathlib import Path
import sys
import pickle
from tqdm import tqdm
import rasterio
from rasterio import sample
import fiona.transform
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, AffineTransformer

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
#    centre = reproject_coords(latlong, orthographic, [[lon_0, lat_0]])[0]
#    top = reproject_coords(latlong, orthographic, [[lon_0, lat_0+box_size/2]])[0]
#    bottom = reproject_coords(latlong, orthographic, [[lon_0, lat_0-box_size/2]])[0]
#    width = top[1]-bottom[1]


    # Use pyproj for faster coordinate transforms
    transformer = Transformer.from_crs(latlong, orthographic, always_xy=True)
    coords = [
        (lon_0, lat_0),
        (lon_0, lat_0 + box_size / 2),
        (lon_0, lat_0 - box_size / 2)
    ]
    nxs, nys = transformer.transform(*zip(*coords))
    centre = (nxs[0], nys[0])
    top = (nxs[1], nys[1])
    bottom = (nxs[2], nys[2])
    width = top[1] - bottom[1]

    
    new_left, new_bottom, new_right, new_top = centre[0]-width/2, bottom[1], centre[0]+width/2, top[1]
    #  want an output raster of width x height pixels:
    dst_width, dst_height = dim,dim
    # Create the affine transform for the destination raster:
    dst_transform = from_bounds(new_left, new_bottom, new_right, new_top, dst_width, dst_height)
    src_transform = src.transform
    src_crs = src.crs


    # Reproject destination bounds from ORTHO to EPSG:3395 (source CRS)
    ortho_to_merc = Transformer.from_crs(orthographic, "EPSG:3395", always_xy=True)
    src_bounds = ortho_to_merc.transform_bounds(new_left, new_bottom, new_right, new_top, densify_pts=10)
    
    # Get pixel bounds in source image
    row_start, col_start = rowcol(src.transform, src_bounds[0], src_bounds[3])  # top-left
    row_stop, col_stop = rowcol(src.transform, src_bounds[2], src_bounds[1])   # bottom-right
    
    # Make sure indices are valid
    row_start, row_stop = sorted((max(0, row_start), min(img.shape[0], row_stop)))
    col_start, col_stop = sorted((max(0, col_start), min(img.shape[1], col_stop)))
    
    # Crop the image more accurately
    d = img[row_start:row_stop, col_start:col_stop]
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    src_transform_cropped = window_transform(window, src.transform)

    dst_transform = from_bounds(new_left, new_bottom, new_right, new_top, dim, dim)

    # Reproject
    dst_data = np.empty((dim, dim), dtype=img.dtype)

    try:
        reproject(
            source=d,
            destination=dst_data,
            src_transform=src_transform_cropped,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=orthographic,
            resampling=Resampling.nearest
        )
    except:
        reproject(
            source=img,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=orthographic,
            resampling=Resampling.nearest
        )

#
#    
#    # Prepare an empty array for the destination data
#    dst_data = np.empty((dst_height, dst_width), dtype=img.dtype)
#    dst_crs = orthographic
#    # # Reproject the data:
#    pp = reproject(
#        source=img,
#        destination=dst_data,
#        src_transform=src_transform,
#        src_crs=src_crs,
#        dst_transform=dst_transform,
#        dst_crs=dst_crs,
#        resampling=Resampling.nearest  # or another resampling method as needed
#    )
#    #Now make circles on the data
    tr = AffineTransformer(dst_transform)
    #samples

    #inverse the transform
    inv_transform = ~dst_transform
    
    angles = [180,225,270,315]#0,45,90,135]
    lines = dict()
    npoints = None
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
                          ix=line_points[1], iy=line_points[0], z=line_samples,
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
    lon_ortho = np.array(lon_ortho).T#list(zip(*lon_ortho))
    lat_ortho = np.array(lat_ortho).T#list(zip(*lat_ortho))
    _lons = reproject_coords(orthographic, latlong, lon_ortho)
    _lats = reproject_coords(orthographic, latlong, lat_ortho)

    meta["min"] = dst_data.min()
    meta["max"] = dst_data.max()
    meta["mean"] = dst_data.mean()
    meta["median"] = np.median(dst_data)
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

    return dst_data,lines,circles,meta,coords


#----
@postprocess.command()
@click.argument("filename",type=Path)
@click.argument("low", type=int)
@click.argument("high", type=int)
@click.option("--negative",default=False, is_flag=True)
@click.option("--prefix",default="postprocessed", type=str)
@click.option("--force",default=False, is_flag=True)
@click.option("--preload",default=False, is_flag=True)
@click.option("--onlymeta",default=False, is_flag=True)
@click.option("--nofilter",default=False, is_flag=True)
def segment(filename,low, high, negative, prefix, force, preload, onlymeta, nofilter):
    pathfilename = Path(filename)
    source = pd.read_csv(pathfilename, index_col=0)

    #filter to reduce number
    th=10
    if not nofilter: #if nofilter then don't filter!
#       source=source[source.duplicates>(th-np.log2(source["Diameter (km)"]))]
        source = source[(source.duplicates>=4)]
    else:
        prefix = f"unfiltered_{prefix}"
        
    print(f"processing {len(source)} entries")

    #load the source file
    if low==0:
        source.to_csv(pathfilename.with_suffix(pathfilename.suffix+".filtered"))
    if high==0:
        high=len(source)
    source = source.sort_values("duplicates",ascending=False)
    filename = Path("/mnt/export/lee/1-Projects/deepcaldera/source_data/gebco/gebco_2023_clipped_to_seamounts_proj.tif")
#    src = rasterio.open(filename)
#    src_data= src.read(1)

    count = 0
    _seg = source.iloc[low:high]
    addon = dict()
    if negative:
        results = output_directory/f"data/{prefix}/negative"
    else:
        results = output_directory/f"data/{prefix}/positive"
    print(f"Saving to {results}")
    results.mkdir(parents=True, exist_ok=True)
    src = None
    if preload:
        src = rasterio.open(filename)
        src_data= src.read(1)
    import time
    start = time.time()
    for irow, row in tqdm(_seg.iterrows(),total=len(_seg)):
        fname=results/f"{irow}.pkl"
        if fname.exists() and not force:
            d = pickle.load(open(fname,'rb'))
            addon[irow]=d["meta"]
        else:
            if src is None:
                src = rasterio.open(filename)
                src_data= src.read(1)
            try:
                ortho, lines,circles, meta,coords = cross_section(row, src_data, src, dim=256)
                addon[irow]=meta

                if onlymeta:
                   #pickle the meta data only
                    d=dict(meta=meta)
                else:
                    d=dict(ortho=ortho, lines=lines, circles=circles, meta=meta, coords=coords, irow=irow)
                pickle.dump(d,open(fname,'wb'))
            except KeyboardInterrupt:
                raise
            except:
                print(f"Error on row {irow}")
    addon_df = pd.DataFrame(addon).T
    df = pd.merge(_seg,addon_df,left_index=True, right_index=True)
    print(df.columns)
    df = df[df.ring_mean < 1000]
    df.to_csv(results/f"{low:05d}_{high:05d}.csv")
    stop = time.time()
    print("Time: {}".format(stop-start))

@postprocess.command()
@click.option("--suffix", default="top_features")
@click.option("--negative",default=False, is_flag=True)
@click.option("--skiphdf",default=False, is_flag=True)
@click.option("--nofilter",default=False, is_flag=True)
@click.option("--prefix",default="postprocessed", type=str)
def final(suffix, negative, skiphdf, nofilter, prefix):
    if nofilter:
        post = output_directory/f"data/unfiltered_{prefix}"
    else:
        post = output_directory/f"data/{prefix}"

    prefix="positive"
    if negative:
        prefix = "negative"
    post = post/prefix

    d1=[]
    print(post)
    for g in post.glob("*0.csv"):
        print(g)
    for p in sorted([g for g in post.glob("*00*.csv")]):
        print(p)
        d1.append(pd.read_csv(p,index_col=0))
    df = pd.concat(d1)
    df.to_csv(post/f"{prefix}_{suffix}.csv")

    if skiphdf:
        return
    import h5py
    with pd.HDFStore(post/f"{prefix}_{suffix}.hdf",'w') as out:
        d=[]
        c=0
        for p in tqdm(df.sort_index().index):
            fname = post/f"{p}.pkl"#sorted(list(post.glob("*.pkl"))):
            c+=1
            #print(p,c)
            pdata = pickle.load(open(fname,"rb"))
            out[f"feature{p}/ortho"] = pd.DataFrame(pdata["ortho"])
            for k,v in pdata["lines"].items():
                _v = pd.DataFrame(v)
                del _v["variable"]
                out[f"feature{p}/lines/angle{k}"] = _v
            for k,v in pdata["circles"].items():
                _v = pd.DataFrame(v)
                n = str(k).replace(".","_")
                out[f"feature{p}/circles/rad{n}"] = _v
            out[f"feature{p}/coords"] = pd.DataFrame(pdata["coords"])
            d.append(pdata)
        
    
    
if __name__ == "__main__":
    postprocess()

