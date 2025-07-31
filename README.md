
Steps

1. Generate a set of images
	lowres/scripts/global.py -> generate many images by sampling the global map and storing samples in files with 1000 samples each
2. run the CNN over each image to identify circles and extract metadata
	scripts/res_predict_model.py cnn-prediction --index=X --prefix=sys_cal --dataset=DEM -> process file ending with index X using the DEM data.
3. run just the circle finder over the image
	scripts/res_predict_model.py make-prediction --index=X --prefix=sys_cal --dataset=DEM -> process file ending with index X using the DEM data.
	(this is the fast step, but requires access to the gpu)
4. merge the csv files into one
	scripts/postprocess.py combine 
5. extra the metadata and resample images for the cross-sections
	python scripts/postprocess.py segment caldera_positive.csv INDEX_LOW INDEX_HIGH --preload --nofilter
6. finally combine the data
       python scripts/postprocess.py final --nofilter
