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

# +
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import geopandas as gpd
from sentinelhub import DataCollection

# +
#[print(band) for band in DataCollection.SENTINEL2_L2A.bands]

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

def plot_image(image, labels=None, bbox=None, marks=None, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    if len(image)<=10: # tiled images
        n = len(image)

        fig, ax = plt.subplots(nrows=n, ncols=1, figsize=(10, 12*n))
    
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
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

            if marks is not None:
                ax.scatter(marks[0], marks[1], zorder=1, alpha=0.5, c='r')

            BBOX = None
            if bbox is not None:
                BBOX = (bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y)
                ax.set_xlim(bbox.min_x, bbox.max_x)
                ax.set_ylim(bbox.min_y, bbox.max_y)

            ax.set_title(labels)
                
            if clip_range is not None:

                ax.imshow(np.clip(image * factor, *clip_range), zorder=0, 
                          extent=BBOX, aspect='equal', **kwargs)
            else:
                ax.imshow(image * factor, zorder=0, 
                          extent=BBOX, aspect='equal', **kwargs)

    #ax[i].set_xticks([])
    #ax[i].set_yticks([])


# +
#marks = [54.198, 38.494]
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
#ax.scatter(marks[0], marks[1], zorder=1, alpha=0.5, c='r', s=100)
#ax.set_xlim(roi_bbox.min_x, roi_bbox.max_x)
#ax.set_ylim(roi_bbox.min_y, roi_bbox.max_y)
#ax.imshow(np.clip(rgb_image * 3.4/255, 0,1), zorder=0, extent=(roi_bbox.min_x, roi_bbox.max_x, roi_bbox.min_y, roi_bbox.max_y), aspect='equal')
#plt.show()     

# +
# Region of Interest
# Define bounding box using http://bboxfinder.com or Google Earth
# Define special markers [LON, LAT] to show on map
# The bounding box in WGS84 coordinate system is (longitude and latitude coordinates of lower left and upper right corners).
box_size = 0.03

# Korpezhe O&G field (Turkmenistan South)
#roi_marks = [54.1977, 38.4939]
#roi_time_interval = '2021-03-30', '2021-03-31'
#roi_time_interval = '2019-08-24', '2019-08-26'

# Goturdepe field (Turkmenistan North)
#roi_marks = [53.743, 39.474]
#roi_time_interval = '2021-04-10', '2021-04-12'

# Algerian Hassi Messaoud oil field on 20 November 2019 
roi_marks = [5.9053, 31.6585]
roi_time_interval = '2019-11-20', '2019-11-21'

# Aliso Canyon gas leak, Oct 23, 2015
#roi_marks = [-118.56414880359367, 34.31490317735722]
#roi_time_interval = '2015-10-23', '2015-12-31'


# +
resolution = 10 # desired resolution parameter of the image in meters
roi_bbox = BBox([roi_marks[0]-box_size, roi_marks[1]-box_size, roi_marks[0]+box_size, roi_marks[1]+box_size], crs=CRS.WGS84)
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
            mosaicking_order='mostRecent', # values:  mostRecent, leastRecent, leastCC (cloud cover)
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
plot_image(rgb_image, labels=['RGB Visual'], bbox=roi_bbox, marks=roi_marks, factor=2.4/255, clip_range=(0,1))

# +
evalscript_bands_CH4 = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                //bands: ["B08","B09","B10","B11","B12"],
                bands: ["B08","B09","B11","B12"],
                units: "DN"
            }],
            output: {
                bands: 5,
                sampleType: "INT16"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B08,
                sample.B09,
                sample.B10,
                sample.B11,
                sample.B12];
    }
"""

resolution = 20 # desired resolution parameter of the image in meters (lowest available for Band 12)
roi_size = bbox_to_dimensions(roi_bbox, resolution=resolution)
roi_size = (250,250)
request = SentinelHubRequest(
    evalscript=evalscript_bands_CH4,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,#2A,
            time_interval=roi_time_interval,
            mosaicking_order='mostRecent', # values:  mostRecent, leastRecent, leastCC (cloud cover)
    )],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=roi_bbox,
    size=roi_size,
    config=config
)
data_bands_CH4 = request.get_data()
# -

b8 = data_bands_CH4[0][:, :, [0]].astype(int)
b9 = data_bands_CH4[0][:, :, [1]].astype(int)
#b10 = data_bands_CH4[0][:, :, [2]].astype(int)
b11 = data_bands_CH4[0][:, :, [3]].astype(int)
b12 = data_bands_CH4[0][:, :, [4]].astype(int)

#plot_image([b8, b9, b10, b11, b12], labels=['Band 8','Band 9','Band 10','Band 11','Band 12'], bbox=roi_bbox, marks=roi_marks, factor=1.4/1e4, clip_range=(0,1))
plot_image([b8, b11, b12], labels=['Band 8','Band 11','Band 12'], bbox=roi_bbox, marks=roi_marks, factor=1.4/1e4, clip_range=(0,1))

# +
# inspiration:  https://gisgeography.com/sentinel-2-bands-combinations/
#h20 = (b8 - b11) / (b8 + b11)
#plot_image(h20, labels=['Water'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)#clip_range=(0,1))
# -

# # Approach using Bands 11 and 12
#
# Inspired by:  section 3.3, https://amt.copernicus.org/articles/14/2771/2021/
#
# ### Image with methane

# Inspired by:  Equation 3, section 3.2, https://amt.copernicus.org/articles/14/2771/2021/
A = np.vstack([b12.ravel(), np.ones(len(b12.ravel()))]).T
c = np.linalg.lstsq(A.squeeze(), b11.ravel(), rcond=None)[0]
cB12 = c[0]*b12+c[1]
mbsp = (cB12 - b11)/b11
plot_image(mbsp, labels=['CH4_mbsp'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)

# ### Reference image without methane

# +
# Inspired by:  section 3.3, https://amt.copernicus.org/articles/14/2771/2021/
# reference image without methane
resolution = 20 # desired resolution parameter of the image in meters (lowest available for Band 12)
roi_size = bbox_to_dimensions(roi_bbox, resolution=resolution)
roi_size = (250,250)
ref_time_interval = '2019-10-06', '2019-10-07'

request_ref = SentinelHubRequest(
    evalscript=evalscript_bands_CH4,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,#SENTINEL2_L1C
            time_interval=ref_time_interval,
            #mosaicking_order='leastCC',
    )],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF)
    ],
    bbox=roi_bbox,
    size=roi_size,
    config=config
)
data_bands_CH4_ref = request_ref.get_data()
b11_ref = data_bands_CH4_ref[0][:, :, [3]].astype(int)
b12_ref = data_bands_CH4_ref[0][:, :, [4]].astype(int)
# -

A_ref = np.vstack([b12_ref.ravel(), np.ones(len(b12_ref.ravel()))]).T
c_ref = np.linalg.lstsq(A_ref.squeeze(), b11_ref.ravel(), rcond=None)[0]
cB12_ref = c_ref[0]*b12_ref+c[1]
mbsp_ref = (cB12_ref - b11_ref)/b11_ref
plot_image(mbsp_ref, labels=['CH4_mbsp_ref'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)

# ### Difference to derive methane components

mbmp = mbsp - mbsp_ref
plot_image(mbmp, labels=['CH4_mbmp'], bbox=roi_bbox, marks=roi_marks, factor=3.4)#, vmax=1)

# ### Gaussian filter to remove geo features

from scipy.ndimage import gaussian_filter
mbmp_1 = mbmp.copy()
mbmp_1 = gaussian_filter(mbmp_1, sigma=3)
#mbmp_1 = gaussian_filter(mbsp, sigma=3) - gaussian_filter(mbsp_ref, sigma=3)#
plot_image(mbmp_1, labels=['CH4_gaussian'], bbox=roi_bbox, marks=roi_marks, factor=5.4)#, vmax=1)

# ### Difference on normalized images

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
b11_n = scaler.fit_transform(b11.squeeze())
b12_n = scaler.transform(b12.squeeze())
tmp = b12_n - b11_n
plot_image(tmp, labels=['Delta normalized'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)

# ### Clip filter

mbmp_1 = mbmp.copy()
mbmp_1[mbmp_1 > .19] = .3
#mbmp_1[mbmp_1 > mbmp_1.mean()] = 1
plot_image(mbmp_1, labels=['CH4_clip'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)

# ### Median filter

from scipy.ndimage import median_filter
mbmp_1 = mbmp.copy()
#mbmp_1 = median_filter(b12, size=1)
mbmp_1 = median_filter(mbsp, size=5) - median_filter(mbsp_ref, size=4)
plot_image(mbmp_1, labels=['CH4_median'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)

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
    
mbmp_1 = mbmp.copy().squeeze()
kernel = inverse_gaussian_kernel(5)
mbmp_1 = convolve(mbmp_1, kernel)
plot_image(mbmp_1, labels=['CH4_inverse_gaussian'], bbox=roi_bbox, marks=roi_marks, factor=5.4)#, vmax=1)
# -

# # Determine "average" reference image

# +
# Finding an "average" reference image without methane
resolution = 20 # desired resolution parameter of the image in meters (lowest available for Band 12)
roi_size = bbox_to_dimensions(roi_bbox, resolution=resolution)
roi_size = (250,250)
ref_range = ['2019-01-01','2019-02-01','2019-03-01','2019-04-01','2019-05-01','2019-06-01','2019-07-01','2019-08-01']

ref_b11 = []
ref_b12 = []
for ref_date in ref_range:
    ref_time_interval = ref_date, f'{ref_date[:-1]}2'

    request_ = SentinelHubRequest(
        evalscript=evalscript_bands_CH4,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,#SENTINEL2_L1C
                time_interval=ref_time_interval,
                #mosaicking_order='leastCC',
        )],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=roi_bbox,
        size=roi_size,
        config=config
    )
    data_bands_CH4_ = request_.get_data()
    b11_ = data_bands_CH4_[0][:, :, [3]].astype(int)
    b12_ = data_bands_CH4_[0][:, :, [4]].astype(int)
    if np.count_nonzero(b11_)>0:
        ref_b11.append(b11_)
    if np.count_nonzero(b12_)>0:
        ref_b12.append(b12_)
# -

b11_mean = np.mean( np.array([ i for i in ref_b11]), axis=0 )
b12_mean = np.mean( np.array([ i for i in ref_b12]), axis=0 )

A_mean = np.vstack([b12_mean.ravel(), np.ones(len(b12_mean.ravel()))]).T
c_mean = np.linalg.lstsq(A_mean.squeeze(), b11_mean.ravel(), rcond=None)[0]
cB12_mean = c_mean[0]*b12_mean+c[1]
mbsp_mean = (cB12_mean - b11_mean)/b11_mean
plot_image(mbsp_mean, labels=['CH4_mbsp_mean'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)

mbmp_mean = mbsp - mbsp_mean
plot_image(mbmp_mean, labels=['CH4_mbmp_mean'], bbox=roi_bbox, marks=roi_marks, factor=3.4)#, vmax=1)

plot_image(mbmp_mean_2, labels=['CH4_mbmp_mean__median'], bbox=roi_bbox, marks=roi_marks, factor=1, vmin=0.1, vmax=0.16)







# # Different experiments

import matplotlib.colors as colors
#ch4_1 = (b12-b11)/b12#/1e4
ch4_1 = (b12-b11)
#ch4_2 = (b12-b11)/b12
#ch4_1 = (1.1*b12-b12.mean())/b11
#ch4_3 = (b12/b11)
#ch4_4 = np.log(b12/b11)
plot_image(ch4_1, labels=['CH4_1'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)
#plot_image(ch4_1, labels=['CH4_1'], bbox=roi_bbox, marks=roi_marks, factor=3.4, clip_range=(0,1))
#plot_image([ch4_1, ch4_2], labels=['CH4_1', 'CH4_2'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)
#plot_image([ch4_1, ch4_2], labels=['CH4_1', 'CH4_2'], bbox=roi_bbox, marks=roi_marks, clip_range=(-1,0))
#plot_image([ch4_1, ch4_2, ch4_3, ch4_4], labels=['CH4_1', 'CH4_2', 'CH4_3', 'CH4_4'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)#clip_range=(0,1))

# +
#[print(x) for x in mbmp_1.squeeze()[115:135,110:130].round(2)]

# +
#[print(x) for x in mbmp.squeeze()[115:135,110:130].round(2)]
#[[f'{z:0.2f}' for z in x] for x in mbmp.squeeze()[115:135,110:130].round(2)]

# +
#(b12-b11).squeeze()[100:120,85:94]

# +
#tmp = ch4_1.clip(-np.inf, -100)
#plot_image(tmp, labels=['CH4'], bbox=roi_bbox, marks=roi_marks, factor=3.4, vmax=1)#clip_range=(0,1))
# -






