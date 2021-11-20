# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (Data Science)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/datascience-1.0
# ---

# # Sentinel 2 data processing
#
# Since its launch in 2018, Sentinel-5P provides directly calculated estimates of CH4 levels, however the resolution is 5.5km x 7.0km.
#
# Recent advances in algorithms enable estimation of CH4 levels from SWIR (Shortwave Infrared) bands measure by prior satellites, such as Sentinel-2, at a much high resolution, 20m x 20m.
#
# Approach:
#
# * Retrieve sentinel-2 data
# * Estimate CH4
#
# Inspired by:
#
# * https://github.com/sentinel-hub/sentinelhub-py/blob/master/examples/data_collections.ipynb
# * https://amt.copernicus.org/preprints/amt-2021-238/amt-2021-238.pdf

# +
# #!pip install sentinelhub
# #!pip install cdsapi

# +
# #!pip install xarray[io]
# #!pip install netcdf4
# #!pip install h5netcdf
# restart kernel !!!

# +
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from tqdm.auto import tqdm
import boto3
import botocore
import datetime
import random
import xarray as xr
from os.path import exists
import os
from dateutil import parser
import datetime

import geopandas as gpd
from sentinelhub import DataCollection
import cdsapi

# +
#[print(band) for band in DataCollection.SENTINEL2_L2A.bands]
# -

from sentinelhub import AwsTile, AwsTileRequest
from botocore.errorfactory import ClientError

# +
# %%time
bucket_name = 'sentinel-s2-l1c'
save_folder = './AwsData'

zones = ['13SER']#, '13SFR','13SGR','13REQ','13RFQ','13RGQ']
bands = ['B11', 'B12']
date_range = {'from': '2017-01-01',
              'to': '2017-01-20'}

aws_index = 0
s3 = boto3.client('s3')

satellite = bucket_name.split('-')[1]
level = bucket_name[-3:]

for zone in tqdm(zones):
    assert len(zone)==5
    zone_f = f'{zone[:2]}/{zone[2:3]}/{zone[-2:]}'
 
    start_date = parser.parse(date_range['from'])
    end_date = parser.parse(date_range['to'])
    delta = end_date - start_date
    
    for d in tqdm(range(delta.days + 1), leave=False):
        
        pull_date = start_date + datetime.timedelta(days=d)
        pull_date_f = f'{pull_date.year}/{pull_date.month}/{pull_date.day}'
        
        for band in bands:
            aws_key = f'tiles/{zone_f}/{pull_date_f}/{aws_index}/{band}.jp2'
            save_path = f'{save_folder}/{satellite}/{level}/{zone}/{pull_date.strftime("%Y-%m-%d")}/'
            save_file = f'{band}.jp2'
            print(aws_key)
            s3.head_object(Bucket=bucket_name, Key=aws_key)
            try:
                os.makedirs(save_path, exist_ok=True)
            except ClientError:
                continue # Not found
            s3.download_file(bucket_name, aws_key, save_path+save_file, ExtraArgs={'RequestPayer': 'requester'})
            
# -







# +
tile_name = '13SER'
time = '2017-01-19'
aws_index = 0

bands = ['B11', 'B12']
metafiles = []#'tileInfo']#, 'preview', 'qi/MSK_CLOUDS_B00']
data_folder = './AwsData'

request = AwsTileRequest(
    tile=tile_name,
    time=time,
    aws_index=aws_index,
    bands=bands,
    metafiles=metafiles,
    data_folder=data_folder,
    data_collection=DataCollection.SENTINEL2_L1C
)

request.save_data(redownload=True)
# -

b11, b12 = request.get_data()
#request.









# +
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, MimeType, bbox_to_dimensions

config = SHConfig() # credentials will default to locally configured AWS credentials
if True: # first time setup
    pass
    #config.sh_client_id = '<sh_oath_id>'
    #config.sh_client_secret = '<sh_oath_secret>'
    #config.aws_access_key_id = '<aws_key_id>'
    #config.aws_secret_access_key = '<aws_key_secret>'
    #config.save()
else:
    config.load()
# -











def plot_image(image, labels=None, bbox=None, marks=None, factor=1.0, clip_range=None, size=None, wind=None, **kwargs):
    width = 5 if size==None else size
    height = width+2
    if len(image)<=10: # tiled images
        n = len(image)
        fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(width, height*n))    
        for i, img, label in zip(range(n), image, labels):
            if marks is not None:
                ax[i].scatter(marks[0], marks[1], zorder=1, alpha=0.5, c='r')
            BBOX = None
            if bbox is not None:
                BBOX = (bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y)
                ax[i].set_xlim(bbox.min_x, bbox.max_x)
                ax[i].set_ylim(bbox.min_y, bbox.max_y)            
            if label is not None:
                ax[i].set_title(label)                
            if clip_range is not None:
                ax[i].imshow(np.clip(img * factor, *clip_range), zorder=0, 
                          extent=BBOX, aspect='equal', **kwargs)
            else:
                ax[i].imshow(img * factor, zorder=0, 
                          extent=BBOX, aspect='equal', **kwargs)       
    else: # single image
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
            if marks is not None:
                ax.scatter(marks[0], marks[1], zorder=1, alpha=0.5, c='r')
            BBOX = None
            if bbox is not None:
                BBOX = (bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y)
                ax.set_xlim(bbox.min_x, bbox.max_x)
                ax.set_ylim(bbox.min_y, bbox.max_y)
            ax.set_title(labels)
            if wind is not None:
                ax.barbs(*wind, zorder=1, length=10, pivot='tip', barbcolor='w', flip_barb=True)
            if clip_range is not None:
                ax.imshow(np.clip(image * factor, *clip_range), zorder=0, 
                          extent=BBOX, aspect='equal', **kwargs)
            else:
                ax.imshow(image * factor, zorder=0, 
                          extent=BBOX, aspect='equal', **kwargs)


def getRandomDates(end_dt, window=3, n=100):
    #sets up array of n random dates for a window of X years
    end_yr, end_m, end_d = [int(x) for x in end_dt.split('-')]
    end_date = datetime.date(end_yr, end_m, end_d)
    start_date = datetime.date(end_yr-window, end_m, end_d)

    days_between_dates = (end_date - start_date).days
    random_dates = []
    for i in range(n):
        random_number_of_days = random.randrange(days_between_dates)
        random_dates.append(start_date + datetime.timedelta(days=random_number_of_days))

    #print(start_date, end_date)
    #[x for x in random_dates]
    
    return random_dates


# +
# Region of Interest
# Define bounding box using http://bboxfinder.com or Google Earth
# Define special markers [LON, LAT] to show on map
# The bounding box in WGS84 coordinate system is (longitude and latitude coordinates of lower left and upper right corners).

location = 3
box_size = 0.03

if location==1:
    # DOES INDICATE METHANE
    roi_name = 'Korpezhe O&G field (Turkmenistan South)'
    roi_marks = [54.1977, 38.4939]
    box_size=0.01
    roi_time_interval = '2019-03-10', '2019-03-11'

elif location==2:
    # DOES NOT INDICATE METHANE
    roi_name = 'Goturdepe field (Turkmenistan North)'
    roi_marks = [53.743, 39.474]
    roi_time_interval = '2021-04-01', '2021-04-12'

elif location==3:
    # DOES INDICATE METHANE
    roi_name = 'Algerian Hassi Messaoud oil field'
    roi_marks = [5.9053, 31.6585]
    box_size=0.005
    roi_time_interval = '2019-11-20', '2019-11-21'

elif location==4:
    # DOES NOT INDICATE METHANE
    roi_name = 'Aliso Canyon gas leak'
    roi_marks = [-118.56414880359367, 34.31490317735722]
    roi_time_interval = '2015-10-23', '2015-12-31'

elif location==5:
    # DOES NOT INDICATE METHANE
    roi_name = 'Goturdepe field (Turkmenistan North)'
    roi_marks = [53.775, 39.462]
    roi_time_interval = '2021-04-01', '2021-04-12'
    
elif location==6:
    # DOES NOT INDICATE METHANE
    roi_name = 'Xiligaocun'
    roi_marks = [112.923, 36.257]
    roi_time_interval = '2021-04-20', '2021-04-30'
    
elif location==7:
    roi_name='Kazan Russia'
    roi_marks = [50.516, 55.961]
    box_size = .1
    roi_time_interval = '2021-06-04', '2021-06-06'
    
elif location==8:
    roi_name='Carlsbad, NM, USA'
    roi_marks = [-103.826, 32.23, -103.67, 32.31]
    roi_time_interval = '2020-06-01', '2020-06-10'

elif location==9:
    roi_name='San Juan vent, USA'
    roi_marks = [-108.3890, 36.7928]
    roi_time_interval = '2018-06-01', '2018-06-10'

elif location==10:
    roi_name='Appin vent, Australia'
    roi_marks = [150.7197, -34.1815]
    roi_time_interval = '2018-02-01', '2018-02-07'

if location==11:
    # DOES INDICATE METHANE
    roi_name = 'Korpezhe O&G field'
    roi_marks = [54.199, 38.499]
    roi_time_interval = '2018-11-08', '2018-11-15'

if location==12:
    roi_name = 'Korpezhe O&G field'
    roi_marks = [54.199, 38.499]
    box_size = .02
    roi_time_interval = '2019-01-13', '2019-01-15'

if location==13:
    roi_name = 'Liaoning China'
    roi_marks = [122.123, 39.647]
    box_size = .1
    roi_time_interval = '2021-10-20', '2021-10-20'


# +
if len(roi_marks) < 3: # if 1 coordinate specified, then use as center point
    roi_bbox = BBox([roi_marks[0]-box_size, roi_marks[1]-box_size, roi_marks[0]+box_size, roi_marks[1]+box_size], crs=CRS.WGS84)
else:
    roi_bbox = BBox(roi_marks, crs=CRS.WGS84) # use bbox provided [SW corner, NE corner] in [LON, LAT]

resolution = 10 # desired resolution parameter of the image in meters
roi_size = bbox_to_dimensions(roi_bbox, resolution=resolution)

evalscript_true_color = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04"]
        }],
        output: {
            bands: 3
        }
    };
}

function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""

request = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,#2A,
            time_interval=roi_time_interval,
            mosaicking_order='leastCC', # values:  mostRecent, leastRecent, leastCC (cloud cover)
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=roi_bbox,
    size=roi_size,
    config=config
)

rgb_image = request.get_data()[0]
plot_image(rgb_image, labels='RGB Visual', bbox=roi_bbox, marks=roi_marks, factor=2.4/255, clip_range=(0,1))

# +
evalscript_bands_CH4 = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B11","B12"],
                units: "DN"
            }],
            output: {
                bands: 5,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B11,
                sample.B12];
    }
"""

resolution = 20 # desired resolution parameter of the image in meters (lowest available for Band 12)
roi_size = bbox_to_dimensions(roi_bbox, resolution=resolution)
#roi_size = (250,250)
request = SentinelHubRequest(
    evalscript=evalscript_bands_CH4,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,#2A,
            time_interval=roi_time_interval,
            mosaicking_order='leastCC', # values:  mostRecent, leastRecent, leastCC (cloud cover)
    )],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=roi_bbox,
    size=roi_size,
    config=config
)
data_bands_CH4 = request.get_data()
#b8 = data_bands_CH4[0][:, :, [0]].astype(int)
#b9 = data_bands_CH4[0][:, :, [1]].astype(int)
#b10 = data_bands_CH4[0][:, :, [2]].astype(int)
b11 = data_bands_CH4[0][:, :, [0]].astype(int)
b12 = data_bands_CH4[0][:, :, [1]].astype(int)
plot_image([b11, b12], labels=['Band 11','Band 12'], bbox=roi_bbox, marks=roi_marks, factor=1.4/1e4, clip_range=(0,1))
# -

# # CH4 processing using Bands 11 and 12
#
# Inspired by:  section 3.3, https://amt.copernicus.org/articles/14/2771/2021/
#
# ### Band 12 Refinement

# Inspired by:  Equation 3, section 3.2, https://amt.copernicus.org/articles/14/2771/2021/
A = np.vstack([b12.ravel(), np.ones(len(b12.ravel()))]).T
c = np.linalg.lstsq(A.squeeze(), b11.ravel(), rcond=None)[0]
cB12 = c[0]*b12+c[1]
mbsp = (cB12 - b11)/b11
plot_image(mbsp, labels=f'{roi_name} {roi_time_interval[0]} CH4_mbsp', bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1, size=15)

# ### Construct "average" reference image

# +
# Construct a reference image without methane by sampling 10 random 
# frames from the prior 3 years and "averaging" them

ref_range = getRandomDates(roi_time_interval[0])
good_img_count = 0
ref_b11 = []
ref_b12 = []

with tqdm(total=10) as pbar:
    while good_img_count < 10:
        ref_date_st = ref_range.pop()
        ref_date_end = ref_date_st+datetime.timedelta(days=5)

        request_ = SentinelHubRequest(
            evalscript=evalscript_bands_CH4,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,#SENTINEL2_L2A
                    time_interval=(str(ref_date_st), str(ref_date_end)),
                    mosaicking_order='leastCC', # values:  mostRecent, leastRecent, leastCC (cloud cover)
            )],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=roi_bbox,
            size=roi_size,
            config=config
        )
        data_bands_CH4_ = request_.get_data()
        b11_ = data_bands_CH4_[0][:, :, [0]].astype(int)
        b12_ = data_bands_CH4_[0][:, :, [1]].astype(int)
        if (np.count_nonzero(b11_)>0) & (np.count_nonzero(b12_)>0):
            ref_b11.append(b11_)
            ref_b12.append(b12_)
            good_img_count+=1
            pbar.update(1)

b11_mean = np.mean( np.array([ i for i in ref_b11]), axis=0 )
b12_mean = np.mean( np.array([ i for i in ref_b12]), axis=0 )
# -

A_mean = np.vstack([b12_mean.ravel(), np.ones(len(b12_mean.ravel()))]).T
c_mean = np.linalg.lstsq(A_mean.squeeze(), b11_mean.ravel(), rcond=None)[0]
cB12_mean = c_mean[0]*b12_mean+c[1]
mbsp_mean = (cB12_mean - b11_mean)/b11_mean
mbmp_mean = mbsp - mbsp_mean
plot_image(mbmp_mean, labels=f'{roi_name} {roi_time_interval[0]} CH4_mbsp_mean', bbox=roi_bbox, marks=roi_marks, factor=3.4, size=5)#, vmax=1)

# ### wind

# +
era5_bucket = 'era5-pds'
client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))
era5_date = datetime.date(*[int(x) for x in roi_time_interval[0].split('-')])

var_list = ["eastward_wind_at_10_metres", "northward_wind_at_10_metres",
           "eastward_wind_at_100_metres", "northward_wind_at_100_metres"]

year = era5_date.strftime('%Y')
month = era5_date.strftime('%m')

for var in tqdm(var_list):
    s3_data_key = f'{year}/{month}/data/{var}.nc'
    local_file = f'weather_bcj/{year}_{month}_{var}.nc'
    if not exists(local_file):
        era5_raw = client.download_file(era5_bucket, s3_data_key, local_file)

# +
retrieve_time = str(era5_date)+'T12:30:00.000Z' # get this from satellite metadata files

if len(roi_marks)<3:
    ctr_lon = roi_marks[0]
    ctr_lat = roi_marks[1]
else:
    ctr_lon = (roi_marks[0]+roi_marks[2])/2
    ctr_lat = (roi_marks[1]+roi_marks[3])/2

ds = xr.open_dataset(f'weather_bcj/{year}_{month}_{var_list[2]}.nc')
wind_u = ds.sel(time0=retrieve_time, lon=ctr_lon, lat=ctr_lat, method='nearest').eastward_wind_at_100_metres.values

ds = xr.open_dataset(f'weather_bcj/{year}_{month}_{var_list[3]}.nc')
wind_v = ds.sel(time0=retrieve_time, lon=ctr_lon, lat=ctr_lat, method='nearest').northward_wind_at_100_metres.values
# -

retrieve_time

wind_data = [ctr_lon, ctr_lat, wind_u, wind_v]
plot_image(mbmp_mean, labels=f'{roi_name} {roi_time_interval[0]} CH4_mbsp_mean', bbox=roi_bbox, marks=roi_marks, wind=wind_data, factor=3.4, size=5)







# ### Gaussian filter to remove geo features

from scipy.ndimage import gaussian_filter
mbmp_1 = mbmp_mean.copy()
mbmp_1 = gaussian_filter(mbmp_1, sigma=3)
#mbmp_1 = gaussian_filter(mbsp, sigma=3) - gaussian_filter(mbsp_ref, sigma=3)#
plot_image(mbmp_1, labels=['CH4_gaussian'], bbox=roi_bbox, marks=roi_marks, factor=5.4)#, vmax=1)

# ### Inverse Gaussian Filter : Custom by BCJ

# +
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

x, y = np.meshgrid(np.linspace(-1,1,250), np.linspace(-1,1,250))
dst = np.sqrt(x*x+y*y)
# Intializing sigma and muu
sigma = .1
muu = 0.01
# Calculating Gaussian array
z = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
z = z.max() - z
ax.scatter3D(x, y, z, c=z, cmap='Greens')

# +
from scipy.ndimage import convolve

def inverse_gaussian_kernel(filter_size):
    x, y = np.meshgrid(np.linspace(-1,1,filter_size), np.linspace(-1,1,filter_size))
    dst = np.sqrt(x*x+y*y)
    sigma = .1
    muu = 0.0
    z = np.exp(-((dst-muu)**2/(2.0*sigma**2)))
    z = z.max() - z
    return z
    
mbmp_1 = mbmp_mean.copy().squeeze()
kernel = inverse_gaussian_kernel(5)
mbmp_1 = convolve(mbmp_1, kernel)
plot_image(mbmp_1, labels=['CH4_inverse_gaussian'], bbox=roi_bbox, marks=roi_marks, factor=5.4)#, vmax=1)
