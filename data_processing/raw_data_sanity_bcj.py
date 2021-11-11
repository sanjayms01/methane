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

# !pip install xarray
# !pip install geopandas
# !pip install shapely
# !pip install netCDF4

# +

import traceback
import sys
import subprocess
import pickle
import os
import glob
from tqdm.auto import tqdm

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

# # Aggregate Monthly Data to One Dataframe, by Year

# +
data_key_list_2018 = [ '2018-11-meth.parquet.gzip',  '2018-12-meth.parquet.gzip']

data_key_list_2019 = [ '2019-01-meth.parquet.gzip',  '2019-02-meth.parquet.gzip', '2019-03-meth.parquet.gzip', '2019-04-meth.parquet.gzip', '2019-05-meth.parquet.gzip',
 '2019-06-meth.parquet.gzip', '2019-07-meth.parquet.gzip', '2019-08-meth.parquet.gzip', '2019-09-meth.parquet.gzip', '2019-10-meth.parquet.gzip', '2019-11-meth.parquet.gzip',
                      '2019-12-meth.parquet.gzip'
                     ]

data_key_list_2020 = [ '2020-01-meth.parquet.gzip', '2020-02-meth.parquet.gzip', '2020-03-meth.parquet.gzip', '2020-04-meth.parquet.gzip', '2020-05-meth.parquet.gzip',
 '2020-06-meth.parquet.gzip', '2020-07-meth.parquet.gzip', '2020-08-meth.parquet.gzip', '2020-09-meth.parquet.gzip', '2020-10-meth.parquet.gzip', '2020-11-meth.parquet.gzip',
                       '2020-12-meth.parquet.gzip'
                     ]

data_key_list_2021 = [ '2021-01-meth.parquet.gzip', '2021-02-meth.parquet.gzip', '2021-03-meth.parquet.gzip', '2021-04-meth.parquet.gzip', '2021-05-meth.parquet.gzip',
 '2021-06-meth.parquet.gzip', '2021-07-meth.parquet.gzip', '2021-08-meth.parquet.gzip', '2021-09-meth.parquet.gzip',
                        # '2021-10-meth.parquet.gzip'
                     ]

# +
# INITIALIZE FIRST DATAFRAME

location_name = 'testsite3'

bucket = 'methane-capstone'
subfolder = 'month-raw-data'
def read_sentinel5p(subsubfolder, data_key):
    s3_path = bucket+f'/data/{location_name}/'+subfolder + '/' + subsubfolder
    data_location = 's3://{}/{}'.format(s3_path, data_key)
    return data_location

#2018
#subsubfolder = '2018'               
#data_key = data_key_list_2018[0]    
#df_2018 = pd.read_parquet(read_sentinel5p(subsubfolder, data_key))  #read

#2020
subsubfolder = '2019'               
data_key = data_key_list_2019[0]    
df_2019 = pd.read_parquet(read_sentinel5p(subsubfolder, data_key))  #read

#2020
subsubfolder = '2020'               
data_key = data_key_list_2020[0]    
df_2020 = pd.read_parquet(read_sentinel5p(subsubfolder, data_key))  #read

#2020
subsubfolder = '2021'               
data_key = data_key_list_2021[0]    
df_2021 = pd.read_parquet(read_sentinel5p(subsubfolder, data_key))  #read

#Describes
print("shape: ", df_2019.shape)
print("types: ", df_2019.dtypes)
df_2019.head()

# +
# CONCATENATING ALL DATA BY YEAR

#for each in data_key_list_2018[1:]:
#    data_key = each
#    #Read
#    df_cur = pd.read_parquet(read_sentinel5p('2018', data_key))
#    #Concat df to large df
#    df_2018 = pd.concat([df_2018,df_cur], ignore_index=True)
    
for each in data_key_list_2019[1:]:
    data_key = each
    #Read
    df_cur = pd.read_parquet(read_sentinel5p('2019', data_key))
    #Concat df to large df
    df_2019 = pd.concat([df_2019,df_cur], ignore_index=True)

for each in data_key_list_2020[1:]:
    data_key = each
    #Read
    df_cur = pd.read_parquet(read_sentinel5p('2020', data_key))
    #Concat df to large df
    df_2020 = pd.concat([df_2020,df_cur], ignore_index=True)

for each in data_key_list_2021[1:]:
    data_key = each
    #Read
    df_cur = pd.read_parquet(read_sentinel5p('2021', data_key))
    #Concat df to large df
    df_2021 = pd.concat([df_2021,df_cur], ignore_index=True)
    
#Describes
#print("2018 shape: ", df_2018.shape)
print("2019 shape: ", df_2019.shape)
print("2020 shape: ", df_2020.shape)
print("2021 shape: ", df_2021.shape)
# -

len(df_2021.time_utc.unique())

# CONCATENATING ALL DATA
df_combined = df_2019
#df_combined = pd.concat([df_combined,df_2019], ignore_index=True)
df_combined = pd.concat([df_combined,df_2020], ignore_index=True)
df_combined = pd.concat([df_combined,df_2021], ignore_index=True)
print("Total shape: ", df_combined.shape)

df_combined.head()

df_combined

# +
#groupby_date

# +
df_combined['date'] = pd.to_datetime(df_combined['time_utc']).dt.date
groupby_date = df_combined.groupby(df_combined['date']).count()

plt.figure(figsize=(40,20))
plt.bar(groupby_date.index, groupby_date.time_utc)
# -

groupby_date.time_utc.describe()

plt.boxplot(groupby_date.time_utc)

# +
#Write the dataframe to 1 parquet file
file_name='combined-raw.parquet'

bucket = 'methane-capstone'
subfolder = 'combined-raw-data'
s3_path_month = bucket+f'/data/{location_name}/'+subfolder

df_combined.to_parquet('s3://{}/{}'.format(s3_path_month,file_name), compression='gzip')

# -

# # Load Data In
#
#

# +

bucket = 'methane-capstone'
subfolder = 'month-raw-data'
s3_path_month = bucket+f'/data/{location_name}/'+subfolder


data_key = '2019-02-meth.parquet.gzip'
data_location = 's3://{}/{}'.format(s3_path_month+'/2019', data_key)
test_df = pd.read_parquet(data_location)
print(test_df.shape)
test_df.head()


# -

test_df.describe()

test_df.methane_mixing_ratio_bias_corrected.hist()

plt.scatter(test_df.lon, test_df.lat)



