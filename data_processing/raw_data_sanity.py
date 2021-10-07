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

# # Imports

# +

import traceback
import sys
import subprocess
import pickle
import os
import glob

from collections import Counter
try:

    from matplotlib import pyplot as plt #viz
    import matplotlib.colors as colors #colors for viz
    import xarray as xr #process NetCDF
    import numpy as np
    import pandas as pd #data manipulation
    import matplotlib.gridspec as gridspec #create subplot
    from glob import iglob #data access in file manager
    from os.path import join 
    from functools import reduce #string manipulation
    import itertools #dict manipulation
    import matplotlib.patches as mpatches
    
    from datetime import datetime, timedelta
    import time
    import pytz
    
    
    #GeoPandas
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    
except ModuleNotFoundError:

    print('\nModule import error', '\n')
    print(traceback.format_exc())

else:
    print('\nAll libraries properly loaded!!', '\n')
# -

# # Load Data In
#
#

# +

bucket = 'methane-capstone'
subfolder = 'month-raw-data'
s3_path_month = bucket+'/'+subfolder


data_key = '2019-01-meth.parquet.gzip'
data_location = 's3://{}/{}'.format(s3_path_month+'/2019', data_key)
test_df = pd.read_parquet(data_location)
print(test_df.shape)
test_df.head()


# -

test_df.describe()

test_df.methane_mixing_ratio_bias_corrected.hist()

plt.scatter(test_df.lon, test_df.lat)

np.arange(1,10)


