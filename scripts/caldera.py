"""Miscellaneous code for the caldera project. 

Most of the code is projection code and new postprocessing code
"""

# import cratertools.metric as metric
# projection tools
from pyproj import Transformer
from rasterio.transform import rowcol, from_bounds, AffineTransformer
from rasterio.windows import Window, transform as window_transform, from_bounds as window_from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.errors import WindowError
import rasterio
from rasterio import sample
import fiona.transform

# general tools
from datetime import datetime
from pathlib import Path
import sys
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_window(src, lon_min, lat_min, lon_max, lat_max):
    # --- 1. Transform your lat/lon bbox into the raster's CRS ---
    src_bounds = transform_bounds(
        "EPSG:4326", src.crs,
        lon_min, lat_min, lon_max, lat_max
    )

    # --- 2. Check intersection with raster bounds BEFORE reading ---
    raster_bounds = src.bounds  # (left, bottom, right, top) in src CRS

    if (src_bounds[0] >= raster_bounds.right  or   # requested left  >= raster right
        src_bounds[2] <= raster_bounds.left    or   # requested right <= raster left
        src_bounds[1] >= raster_bounds.top     or   # requested bottom >= raster top
        src_bounds[3] <= raster_bounds.bottom):     # requested top   <= raster bottom
        raise ValueError(f"Requested bbox {src_bounds} does not intersect raster {raster_bounds}")

    # --- 3. Compute window, catching any floating point edge cases ---
    try:
        window = window_from_bounds(*src_bounds, transform=src.transform)
        window = window.intersection(
            Window(0, 0, src.width, src.height)  # clamp to valid pixels
        )
    except WindowError as e:
        raise ValueError(f"Bounds are adjacent to but don't overlap the raster: {e}")

    return window

def reproject_coords(src_crs, dst_crs, coords):
    """Reprojects coordinates from the src (e.g. mercator) to dst (plate_caree) using Fiona

    Args:
        src_crs (string): A string description of the source map projection
        dst_crs (string): destination map projection
        coords (iterable): coordinations in x,y pairs

    Returns:
        array: reprojected coordinates
    """
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    nxs, nys = fiona.transform.transform(src_crs, dst_crs, xs, ys)
    return [[x,y] for x,y in zip(nxs, nys)]

def cross_section(row, img, src, dim=256, row2=None):
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
    # convert from lat long to ortho
    lon_0, lat_0, diameter_km = row.Long, row.Lat, row["Diameter (km)"]
    box_size_km = 3 * diameter_km
    planet_rad_km = 6371
    km_to_deg = 360 / (2 * np.pi * planet_rad_km * np.cos(np.deg2rad(row.Lat)))
    box_size_deg = km_to_deg * box_size_km  # approx box
    box_size = box_size_deg

    ##1. extract the ortho image
    coords = [[lon_0, lat_0]]

    mercator = "EPSG:3395"
    latlong = "EPSG:4326"
    orthographic = dict(proj="ortho", lat_0=lat_0, lon_0=lon_0)

    # Use pyproj for faster coordinate transforms
    transformer = Transformer.from_crs(latlong, orthographic, always_xy=True)
    coords = [
        (lon_0, lat_0),
        (lon_0, lat_0 + box_size / 2),
        (lon_0, lat_0 - box_size / 2),
    ]
    nxs, nys = transformer.transform(*zip(*coords))
    centre = (nxs[0], nys[0])
    top = (nxs[1], nys[1])
    bottom = (nxs[2], nys[2])
    width = top[1] - bottom[1]

    new_left, new_bottom, new_right, new_top = (
        centre[0] - width / 2,
        bottom[1],
        centre[0] + width / 2,
        top[1],
    )
    #  want an output raster of width x height pixels:
    dst_width, dst_height = dim, dim
    # Create the affine transform for the destination raster:
    dst_transform = from_bounds(
        new_left, new_bottom, new_right, new_top, dst_width, dst_height
    )
    src_transform = src.transform
    src_crs = src.crs

    dst_transform = from_bounds(new_left, new_bottom, new_right, new_top, dim, dim)


    
    if src.crs.to_string()=="EPSF:3395" : #MERCATOR
        # Reproject destination bounds from ORTHO to EPSG:3395 (source CRS)
        ortho_to_merc = Transformer.from_crs(orthographic, "EPSG:3395", always_xy=True)
        src_bounds = ortho_to_merc.transform_bounds(
            new_left, new_bottom, new_right, new_top, densify_pts=10
        )

        # Get pixel bounds in source image
        row_start, col_start = rowcol(
            src.transform, src_bounds[0], src_bounds[3]
        )  # top-left
        row_stop, col_stop = rowcol(
            src.transform, src_bounds[2], src_bounds[1]
        )  # bottom-right

        # Make sure indices are valid
        row_start, row_stop = sorted((max(0, row_start), min(img.shape[0], row_stop)))
        col_start, col_stop = sorted((max(0, col_start), min(img.shape[1], col_stop)))

        # Crop the image more accurately
#        print(row_start, row_stop, col_start, col_stop)
        data = img[row_start:row_stop, col_start:col_stop]
        window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
        src_transform_cropped = window_transform(window, src.transform)
        dst_data = np.empty((dim, dim), dtype=data.dtype)
    elif src.crs.to_string()=="ESRI:54034":
 #       print("54035")
        lon_min, lon_max = lon_0 - box_size / 2,lon_0 + box_size / 2
        lat_min, lat_max =  lat_0 - box_size / 2, lat_0 + box_size / 2

        src_crs = src.crs
#        print("transform")
        src_bounds = transform_bounds("EPSG:4326", src_crs, lon_min, lat_min, lon_max, lat_max)

        try:
            window = load_window(src, lon_min, lat_min, lon_max, lat_max)
        except ValueError as e:
            print(f"Error occurred while loading window: {e}")
            print(lon_min, lat_min, lon_max, lat_max)
            if row2 is not None:
                return None,None,None,None,None,1,None
            else:
                return None,None,None,None,None,1
#            return None, False
        
#        print("window")
        if window is None:
            print("WINDOW IS NONE?", lon_min, lat_min, lon_max, lat_max)
            if row2 is not None:
                return None,None,None,None,None,1,None
            else:
                return None,None,None,None,None,1
#        print("window2?")
        window = window_from_bounds(*src_bounds, src.transform)
#        print("window3?")
        window = window.intersection(
            Window(0, 0, src.width, src.height)
        )
#        print("read data: " )
        data = src.read(window=window)                    # shape: (bands, rows, cols)
        src_transform_cropped = src.window_transform(window)      # <-- correct origin for this window
#        print("almost in ortho")
    elif src.crs.to_string()=="EPSG:4236" : # PLATE CAREE
        data = img
        src_transform_cropped=src.transform

    dst_data = np.empty((dim, dim), dtype=data.dtype)
        
    try:
        reproject(
            source=data,
            destination=dst_data,
            src_transform=src_transform_cropped,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=orthographic,
            resampling=Resampling.nearest,
        )
    except:
        reproject(
            source=img,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=orthographic,
            resampling=Resampling.nearest,
        )
#    print("DONE TRANSFORM")
    #
    #
    #    #Now make circles on the data
    tr = AffineTransformer(dst_transform)
    # samples

    # inverse the transform
    inv_transform = ~dst_transform

    angles = [180, 225, 270, 315]  # 0,45,90,135]
    lines = dict()
    npoints = None
    # lines
    rad = diameter_km * 1e3 / 2
    x0, y0 = 0, 0  # in the image coords

    def collect_points(points):
        image_points = inv_transform * points
        row_f = np.array(image_points[0], dtype=float)
        col_f = np.array(image_points[1], dtype=float)
    
        # build a validity mask: must be finite and within array bounds
        valid = (
            np.isfinite(row_f) & np.isfinite(col_f) &
            (row_f >= 0) & (row_f < dst_data.shape[0]) &
            (col_f >= 0) & (col_f < dst_data.shape[1])
        )
    
        row = np.where(valid, np.round(row_f).astype(np.int64), 0)
        col = np.where(valid, np.round(col_f).astype(np.int64), 0)
    
        image_samples = np.where(valid, dst_data[row, col], np.nan)
        row_f, col_f = np.array(image_points[0], float), np.array(image_points[1], float)
        if not np.all(np.isfinite(row_f)):
            print(f"  non-finite rows: {np.sum(~np.isfinite(row_f))} / {len(row_f)}")
        image_samples = np.where(dst_data[row,col]<1e5, dst_data[row, col], np.nan)
        return np.array(image_points), image_samples

    def safe_mean(z):
        v = z[np.isfinite(z)]
        return float(np.mean(v)) if len(v) > 0 else np.nan

    def safe_std(z):
        v = z[np.isfinite(z)]
        if len(v) <= 1:
            #print(f"  safe_std WARNING: only {len(v)} finite values in '{label}'")
            return np.nan
    
        return float(np.std(v)) if len(v) > 0 else np.nan


    for ang in angles:
        updown = 0
        peaked = False
#        print("what")
        ath = np.deg2rad(ang)
        if npoints is None:
            npoints = 64
        radii = np.linspace(-rad * 2, rad * 2, npoints)
        # sample points
        points = x0 + radii * np.cos(ath), y0 + radii * np.sin(ath)
#        print("collect points")
        line_points, line_samples = collect_points(points)
#        print("done")
        lines[ang] = dict(
            x=points[0],
            y=points[1],
            r=radii,
            ix=line_points[1],
            iy=line_points[0],
            z=line_samples,
            variable=safe_std(line_samples),
        )
#    print("circle loop")
    # circles
    circles = dict()
    for circrad in [0.1, 0.75, 1.0, 1.25]:
        Circpoints = np.linspace(0, 2 * np.pi, npoints)
        cc, sc = circrad * rad * np.cos(Circpoints), circrad * rad * np.sin(Circpoints)
        c_points, c_samples = collect_points((x0 + cc, y0 + sc))
        circles[circrad] = dict(
            ix=c_points[0], iy=c_points[1], z=c_samples, rad=circrad
        )
        if circrad==1.0:
            fracbad = np.sum(np.isnan(c_samples))/c_samples.size
        

    if row2 is not None:
        inner_circles=dict()
        x0,y0 = transformer.transform([row2.Long],[row2.Lat])
#        print(row2.Long, row2.Lat, row.Long,row.Lat)
#        print(x0,y0)
        # x0,y0 = dst_transform * loc_ortho
        for circrad in [0.1,0.75,1.0,1.25]:
            Circpoints = np.linspace(0,2*np.pi,npoints)
            cc,sc =  circrad*rad*np.cos(Circpoints), circrad*rad*np.sin(Circpoints)
            c_points, c_samples = collect_points((x0+cc, y0+sc))
            inner_circles[circrad] = dict(ix=c_points[0], iy=c_points[1],z=c_samples, rad=circrad,cc=cc,sc=sc)

    points = np.meshgrid(np.arange(256), np.arange(256))
    ipoints = dst_transform * points
    dst_feat=np.ma.array(dst_data, mask=[ipoints[0]**2+ipoints[1]**2 >= (rad)**2])

    meta = dict()
    lon_ortho = dst_transform * (np.arange(dst_width), np.zeros(dst_height))
    lat_ortho = dst_transform * (np.zeros(dst_width), np.arange(dst_height))
    lon_ortho = np.array(lon_ortho).T  # list(zip(*lon_ortho))
    lat_ortho = np.array(lat_ortho).T  # list(zip(*lat_ortho))
    _lons = reproject_coords(orthographic, latlong, lon_ortho)
    _lats = reproject_coords(orthographic, latlong, lat_ortho)
    dst_data = np.ma.array(dst_data, mask=dst_data>1e9)
    meta["min"] = dst_data.min()
    meta["max"] = dst_data.max()
    meta["mean"] = dst_data.mean()
    meta["median"] = np.ma.median(dst_data)

    
    for key, rad in [("ring", 1.0), ("centre", 0.1), ("inner", 0.75), ("outer", 1.25)]:
        z = circles[rad]["z"]
        meta[f"{key}_mean"] = safe_mean(z)
        meta[f"{key}_std"]  = safe_std(z)
    
    meta["ridge"] = safe_mean(circles[0.75]["z"]) > safe_mean(circles[1.25]["z"])
    meta["peak"]  = safe_mean(circles[0.1]["z"])  > safe_mean(circles[0.75]["z"])

    coords = dict(lons=[x for x, y in _lons], lats=[y for x, y in _lats])
    # There HAS to be a better way

    if row2 is not None:
        return dst_data, lines, circles, meta, coords, fracbad, inner_circles
    else:
        return dst_data, lines, circles, meta, coords, fracbad

#def cross_section(row, img, src, dim=256,row2=None):
#    """Creates an orthographic projection from a plate caree projection.
#
#    Args:
#        row (pandas.series): the row object (Series) containing the information for the caldera
#        img (array): The original image in plate carree coordinates to project from.
#        src (_type_): _description_
#        dim (int, optional):  The width/height of the output image.  Only square outputs are supported. Defaults to 256.
#        row2 (_type_, optional): An optional second row (e.g. from a nearby crater) to overlay on the first image, used for a better reprojection of the overlapping crater. Defaults to None.
#    
#    Returns:
#        ortho (numpy.ndarray) : The orthographic projection.
#    """
#
#    # convert from lat long to ortho
#    lon_0, lat_0, diameter_km = row.Long, row.Lat, row["Diameter (km)"]
#    box_size_km= 3*diameter_km*np.cos(np.deg2rad(row.Lat)) #image is 3 times larger than the crater, so all images look the same!
#    planet_rad_km = 6371 #radius in km. this is all approximate here.
#    km_to_deg = 360/(2*np.pi*planet_rad_km * np.cos(np.deg2rad(row.Lat)))
#    box_size_deg = km_to_deg * box_size_km #approx box
#    box_size = box_size_deg
#
#    ##1. extract the ortho image
#    coords = [[lon_0, lat_0]]
#
#    mercator = "EPSG:3395" #source image is mercator
#    latlong = "EPSG:4326" #alternate for the real coordinates is lat-long / plate caree
#    orthographic = dict(proj="ortho", lat_0=lat_0,lon_0=lon_0) #destination image is orthographic
#    
#    # Use pyproj for faster coordinate transforms
#    #from plate-caree to orthographic to get the image locations
#    transformer = Transformer.from_crs(latlong, orthographic, always_xy=True)
#    coords = [
#        (lon_0, lat_0),
#        (lon_0, lat_0 + box_size / 2),
#        (lon_0, lat_0 - box_size / 2)
#    ]
#    nxs, nys = transformer.transform(*zip(*coords))
#    centre = (nxs[0], nys[0])
#    top = (nxs[1], nys[1])
#    bottom = (nxs[2], nys[2])
#    width = top[1] - bottom[1]
#
#    new_left, new_bottom, new_right, new_top = (centre[0]-width/2,
#                                                bottom[1],
#                                                centre[0]+width/2,
#                                                top[1])
#
#    #  want an output raster of width x height pixels:
#    dst_width, dst_height = dim,dim
#    # Create the affine transform for the destination raster:
#    dst_transform = from_bounds(new_left, new_bottom, new_right, new_top, dst_width, dst_height)
#    src_transform = src.transform
#    src_crs = src.crs
#
#    #Now we have the destination square, project that to the source square to find the pixels to sample.
#    #This ensures the output is full of valid pixels, even if the source is sampled outside of the original bounds
#    # Reproject destination bounds from ORTHO to EPSG:3395 (source CRS)
#    ortho_to_merc = Transformer.from_crs(orthographic, "EPSG:3395", always_xy=True)
#    src_bounds = ortho_to_merc.transform_bounds(new_left, new_bottom, new_right, new_top, densify_pts=10)
#
#    # Get pixel bounds in source image
#    row_start, col_start = rowcol(src.transform, src_bounds[0], src_bounds[3])  # top-left
#    row_stop, col_stop = rowcol(src.transform, src_bounds[2], src_bounds[1])   # bottom-right
#
#    # Make sure indices are valid
#    row_start, row_stop = sorted((max(0, row_start), min(img.shape[0], row_stop)))
#    col_start, col_stop = sorted((max(0, col_start), min(img.shape[1], col_stop)))
#
#    # Crop the image more accurately since we probably sampled outside of the requested bounds
#    d = img[row_start:row_stop, col_start:col_stop]
#    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
#    src_transform_cropped = window_transform(window, src.transform)
#
#    dst_transform = from_bounds(new_left, new_bottom, new_right, new_top, dim, dim)
#
#    # Reproject
#    dst_data = np.empty((dim, dim), dtype=d.dtype)
#
#    #Try a cropped transform first for speed. Sometimes that fails with an out of bounds issue
#    #if that happens, fallback to the full global image.
#    try:
#        reproject(
#            source=d,
#            destination=dst_data,
#            src_transform=src_transform_cropped,
#            src_crs=src.crs,
#            dst_transform=dst_transform,
#            dst_crs=orthographic,
#            resampling=Resampling.nearest
#        )
#    except:
#        reproject(
#            source=img,
#            destination=dst_data,
#            src_transform=src.transform,
#            src_crs=src.crs,
#            dst_transform=dst_transform,
#            dst_crs=orthographic,
#            resampling=Resampling.nearest
#        )
#
#    #Now make circles on the data
#    tr = AffineTransformer(dst_transform)
#    # samples
#
#    # inverse the transform
#    inv_transform = ~dst_transform
#    
#    #sample angles, starting from east I think
#    angles = [180,225,270,315]
#    lines = dict()
#    npoints = None
#    
#    # lines
#    rad = diameter_km *1e3 / 2
#    x0, y0 = 0,0 #in the image coords
#    def collect_points(points):
#        image_points = inv_transform * points
#        row,col = np.round(image_points[0]).astype(int),np.round(image_points[1]).astype(int)
#        image_samples = dst_data[row,col]
#        return image_points, image_samples
#
#    for ang in angles:
#        updown = 0
#        peaked = False
#
#        ath = np.deg2rad(ang)
#        if npoints is None:
#            npoints = 64
#        radii = np.linspace(-rad*2,rad*2, npoints)
#        # sample points
#        points = x0+radii*np.cos(ath),y0+radii*np.sin(ath)
#        line_points, line_samples = collect_points(points)
#
#        lines[ang] = dict(x=points[0],y=points[1],r=radii,
#                          ix=line_points[1], iy=line_points[0], z=line_samples,
#                          variable=np.std(line_samples),
#                         )
#    # circles at varius radii 
#    circles=dict()
#    for circrad in [0.1,0.75,1.0,1.25]:
#        Circpoints = np.linspace(0,2*np.pi,npoints)
#        cc,sc =  circrad*rad*np.cos(Circpoints), circrad*rad*np.sin(Circpoints)
#        c_points, c_samples = collect_points((x0+cc, y0+sc))
#        circles[circrad] = dict(ix=c_points[0], iy=c_points[1],z=c_samples, rad=circrad,cc=cc,sc=sc)
#
#    #if a second row is provided, sample that circle too in the current projection. 
#    # These circles don't need to be centered on the image anymore (e.g. a positive detection inside a negative)
#
#    if row2 is not None:
#        inner_circles=dict()
#        x0,y0 = transformer.transform([row2.Long],[row2.Lat])
#        print(row2.Long, row2.Lat, row.Long,row.Lat)
#        print(x0,y0)
#        # x0,y0 = dst_transform * loc_ortho
#        for circrad in [0.1,0.75,1.0,1.25]:
#            Circpoints = np.linspace(0,2*np.pi,npoints)
#            cc,sc =  circrad*rad*np.cos(Circpoints), circrad*rad*np.sin(Circpoints)
#            c_points, c_samples = collect_points((x0+cc, y0+sc))
#            inner_circles[circrad] = dict(ix=c_points[0], iy=c_points[1],z=c_samples, rad=circrad,cc=cc,sc=sc)
#
#    points = np.meshgrid(np.arange(256), np.arange(256))
#    ipoints = dst_transform * points
#    dst_feat=np.ma.array(dst_data, mask=[ipoints[0]**2+ipoints[1]**2 >= (rad)**2])
#    
#    #store the metadata in dictionary
#    meta = dict()
#    lon_ortho = dst_transform*(np.arange(dst_width), np.zeros(dst_height))
#    lat_ortho = dst_transform*(np.zeros(dst_width), np.arange(dst_height))
#    lon_ortho = np.array(lon_ortho).T
#    lat_ortho = np.array(lat_ortho).T
#    _lons = reproject_coords(orthographic, latlong, lon_ortho)
#    _lats = reproject_coords(orthographic, latlong, lat_ortho)
#
#    meta["feat_min"] = dst_feat.min()
#    meta["feat_mean"] = dst_feat.mean()
#    meta["feat_max"] = dst_feat.max()
#    meta["feat_median"] = np.ma.median(dst_feat)
#    meta["min"] = dst_data.min()
#    meta["max"] = dst_data.max()
#    meta["mean"] = dst_data.mean()
#    meta["median"] = np.median(dst_data)
#    meta["ring_mean"] = circles[1.0]["z"].mean()
#    meta["centre_mean"] = circles[0.1]["z"].mean()
#    meta["inner_mean"] = circles[0.75]["z"].mean()
#    meta["outer_mean"] = circles[1.25]["z"].mean()
#
#    meta["ring_std"] = circles[1.0]["z"].std()
#    meta["centre_std"] = circles[0.1]["z"].std()
#    meta["inner_std"] = circles[0.75]["z"].std()
#    meta["outer_std"] = circles[1.25]["z"].std()
#    meta["dst"] = dst_transform
#    meta["inv"] = inv_transform
#
#    meta["ridge"] = np.nanmean(circles[0.75]["z"])>np.nanmean(circles[1.25]["z"])
#    meta["peak"] = np.nanmean(circles[0.1]["z"])>np.nanmean(circles[0.75]["z"])
#
#    coords = dict(lons=[x for x,y in _lons],lats=[y for x,y in _lats])
#
#    #return the appropriate data
#    if row2 is not None:
#        return dst_data,lines,circles,meta,coords, inner_circles
#    else:
#        return dst_data,lines,circles,meta,coords

def plot_features(row, src_data,src, dim=256,row2=None, row3=None, axs=None,second=False,title=None, threshold_bad=0.5):
    """Plots the crater features given the row to plot, global image, and other options.
    Also plots the cross-sections.

    Args:
        row (Series): pandas series
        src_data (numpy.ndarray): source array
        src (geotiff data): src crs information
        dim (int, optional): image size. Defaults to 256.
        row2 (_type_, optional): second caldera to plot. Defaults to None.
        axs (_type_, optional): optional subplot information. Defaults to None.
        second (bool, optional): flag to control plotting. Defaults to False.
        title (_type_, optional): title to plot in the centre of two images. Defaults to None.
    """    
    
    #collect the cross section data
    if row2 is not None:
        ortho, lines, circles, meta, coords, fracbad, circles2 = cross_section(row, src_data, src, dim=256, row2=row2)
        circles2=dict((k,pd.DataFrame(v)) for k,v in circles2.items())
    else:
        ortho, lines, circles, meta, fracbad, coords = cross_section(row, src_data, src, dim=256, row2=row2)
        circles2=None

    if row3 is not None:
        _, _, _, _, _, _, circles3 = cross_section(row, src_data, src, dim=256, row2=row3)
        circles3=dict((k,pd.DataFrame(v)) for k,v in circles3.items())
    else:
        circles3=None


    #if there's no axis provided, generate one.

    circles=dict((k,pd.DataFrame(v)) for k,v in circles.items())
    lines=dict((k,pd.DataFrame(v)) for k,v in lines.items())
    coords = pd.DataFrame(coords)
    feat = dict(ortho=ortho,
                lines=lines,
                circles=circles,
                meta=meta,
                coords=coords,
                circles2=circles2,
                circles3=circles3)

    if row2 is not None:
        #collect row 2 data separately for the metadata.
        ortho, lines, circles, meta, coords, fracbad2= cross_section(row2, src_data, src, dim=256)
        lines=dict((k,pd.DataFrame(v)) for k,v in lines.items())
        coords = pd.DataFrame(coords)
        circles=dict((k,pd.DataFrame(v)) for k,v in circles.items())
        feat2 = dict(ortho=ortho,lines=lines, circles=circles, meta=meta, coords=coords)

    feat["fracbad"] = fracbad
    if fracbad > threshold_bad:
        feat["plotted"]=False
        return feat
    feat["plotted"]=True
    if axs is None:
        fig,axs = plt.subplots(1,2,figsize=(8,4))

    #axis[0] -> plot the image, and overlay lines and circles
    #axis[1] -> plot the height information
    axs[0].imshow(feat["ortho"],
                  vmin=np.min(feat["ortho"][64:192,64:192]),
                  vmax=np.max(feat["ortho"][64:192,64:192]))
    for k,line in feat["lines"].items():
        axs[0].plot(line["ix"], line["iy"])
        axs[1].plot(line.r/1e3, line.z/1e3)
    
    circlek=1.0
    circle = feat["circles"][circlek]
    axs[0].plot(circle["ix"], circle["iy"])
    rad = 3*(circle.ix-128)/circle.ix.max()
    next_color = axs[1]._get_lines.get_next_color()
    
    #plot the locations of the caldera edges.
    axs[1].plot([-circle.rad.iloc[0]*row["Diameter (km)"]/2,
                    circle.rad.iloc[0]*row["Diameter (km)"]/2],
                    np.ones(2)*np.mean(circle.z/1e3),
                    ls='-',color=next_color)
    
    axs[1].plot(np.ones(2)*(-circle.rad.iloc[0]*row["Diameter (km)"]/2),
                [np.min(circle.z/1e3),np.max(circle.z/1e3)],ls='--',color=next_color)
    axs[1].plot(np.ones(2)*(circle.rad.iloc[0]*row["Diameter (km)"]/2),
                [np.min(circle.z/1e3),np.max(circle.z/1e3)],ls='--',color=next_color)
    
    #If row2 is provided, plot that circle too.
    if row2 is not None:
        circlek=1.0
        circle = feat2["circles"][circlek]
        map_scale = row2["Diameter (km)"]/row["Diameter (km)"]
        rad = 3*(circle.ix-128)/circle.ix.max()

        axs[0].plot(128+(circle["ix"]-128)*map_scale, 128+(circle["iy"]-128)*map_scale)
        next_color = axs[1]._get_lines.get_next_color()
        axs[1].plot([-circle.rad.iloc[0]*row2["Diameter (km)"]/2,
                        circle.rad.iloc[0]*row2["Diameter (km)"]/2],
                        np.ones(2)*np.mean(circle.z/1e3),
                        ls='-',color=next_color)
        axs[1].plot(np.ones(2)*(-circle.rad.iloc[0]*row2["Diameter (km)"]/2),
                    [np.min(circle.z/1e3),np.max(circle.z/1e3)],
                    ls='--',color=next_color)
        axs[1].plot(np.ones(2)*(circle.rad.iloc[0]*row2["Diameter (km)"]/2),
                    [np.min(circle.z/1e3),np.max(circle.z/1e3)],
                    ls='--',color=next_color)

    diam = row["Diameter (km)"]
    
    #Now pretty format for the ticks. imshow doesn't do tick values so we format them manually
    lat = row["Lat"]
    lon = row["Long"]
    tv = slice(0,256,50)
    x = feat["coords"].loc[tv,"lons"]
    if np.max(x)-np.min(x) < 0.6:
        fmt = "0.2f"
    elif  np.max(x)-np.min(x) < 6:
        fmt = "0.1f"
    else:
        fmt = "0.1f"
    
    axs[0].set_xticks(np.arange(tv.start,tv.stop,tv.step),
                        labels=[format(f,fmt) for f in feat["coords"].loc[tv,"lons"]],
                        rotation=45)
    x = feat["coords"].loc[tv,"lats"]
    if np.max(x)-np.min(x) < 0.6:
        fmt = "0.2f"
    elif  np.max(x)-np.min(x) < 6:
        fmt = "0.1f"
    else:
        fmt = "0.1f"
    
    axs[0].set_yticks(np.arange(tv.start,tv.stop,tv.step),
                        labels=[format(f,fmt) for f in feat["coords"].loc[tv,"lats"]],
                        rotation=45)
    if title is None:
        title=f"Lon={lon:0.3f}, Lat={lat:0.3f}"
    # Add axis labels and titles
    plt.setp(axs[0], xlabel="Longitude", ylabel="Latitude", title=title)
    plt.setp(axs[1], xlabel="Distance (km)", ylabel="Depth (km)", title=f"Diameter = {diam:0.2f}km")
    plt.tight_layout()
    # plot the extra features if the "second" flag is set. rarely used
    if row2 is not None and second:
        plot_features(row2, src_data,src, dim=dim)

    return feat
