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

# # Pre-Process Data 
#

# +
import traceback
import sys
import subprocess
import pickle
import os
import glob
import datetime
import boto3
import botocore
import datetime

from collections import Counter
try:
    import argparse
    import glob
    import sklearn
    import warnings
    warnings.filterwarnings('ignore')
    import plotly.express as px
    from descartes import PolygonPatch
    from matplotlib import pyplot as plt #viz
    import matplotlib.colors as colors #colors for viz
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import xarray as xr #process NetCDF
    import numpy as np
    import pandas as pd #data manipulation
    import matplotlib.gridspec as gridspec #create subplot
    from glob import iglob #data access in file manager
    from os.path import join 
    from functools import reduce #string manipulation
    import itertools #dict manipulation
    import matplotlib.patches as mpatches
    import geojson
    import json
    import altair as alt 

    from datetime import datetime, timedelta
    import time
    import pytz
    from tqdm import tqdm

    
    #GeoPandas
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    
except ModuleNotFoundError:

    print('\nModule import error', '\n')
    print(traceback.format_exc())

else:
    print('\nAll libraries proeprly loaded!!', '\n')

### HELPER FUNCTIONS
def getHumanTime(seconds):
    '''Make seconds human readable'''
    if seconds <= 1.0:
        return '00:00:01'
    
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '{:d}:{:02d}:{:02d}'.format(h, m, s) # Python 3


def print_write(content, f_object, should_print=True):
    '''This function will both, and write the content to a local file'''
    if should_print: print(content)
    if f_object:
        f_object.write(content)
        f_object.write('\n')
        
alt.data_transformers.disable_max_rows()

# +
#Specify time range of data. Each loop will search for weather for each date. 

from datetime import timedelta, date

start_dt = date(2020, 12, 30 )
end_dt = date(2021, 1, 2)
date_batches = []

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1+timedelta(n)

for dt in daterange(start_dt, end_dt):
    date_batches.append(dt.strftime("%Y-%m-%d"))

print(date_batches)

# -

# ### Operations to apply
#
# * https://heartbeat.comet.ml/working-with-geospatial-data-in-machine-learning-ad4097c7228d
#
#
# * `reading_count` - When aggregating over a given spot, number of readings we have for that spot
# * instead of every day, change data granularity to every week, or every 10 days etc. this would make missing data problem "dissappear"
# * For row (if day then also include week, if week then also include month granularity)
# * potentially instead of lat/lon we can do x,y,z... (may be unnecessary since we are just focussed on CA)

# ### Process data by resolution and time!
#
# * note right now we are not doing anything with `qa_val`

# +
def get_time_resolution_groups(df):
    
    bins = []
    groups = []
    df_reduced = df.groupby(df.time_utc.dt.date).agg({'methane_mixing_ratio': ["count","mean"],
                                                        'lat': ["mean"],
                                                        'lat': ["mean"],
                                                        'methane_mixing_ratio_precision':"mean",
                                                        'methane_mixing_ratio_bias_corrected': "mean",
                                                        'air_pressure_at_mean_sea_level': ["mean"],
                                                        'air_temperature_at_2_metres': ["mean"],
                                                        'air_temperature_at_2_metres_1hour_Maximum': ["mean"],
                                                        'air_temperature_at_2_metres_1hour_Minimum': ["mean"],
                                                        'dew_point_temperature_at_2_metres': ["mean"],
                                                        'eastward_wind_at_100_metres': ["mean"],
                                                        'eastward_wind_at_10_metres': ["mean"],
                                                        'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation' : ["mean"],
                                                        'lwe_thickness_of_surface_snow_amount': ["mean"],
                                                        'northward_wind_at_100_metres': ["mean"],
                                                        'northward_wind_at_10_metres': ["mean"],
                                                        'precipitation_amount_1hour_Accumulation': ["mean"],
                                                        'snow_density': ["mean"],
                                                        'surface_air_pressure': ["mean"],
                                                        'qa_val': [('mode',lambda x:x.value_counts().index[0]), "mean"], #Numerical = Weight, Categorical = Mode
                                                       })

    #Flatten MultiIndex
    df_reduced.columns = ['_'.join(col) for col in df_reduced.columns.values]
    df_reduced = df_reduced.reset_index()
    df_reduced = df_reduced.rename(columns={"methane_mixing_ratio_count": "reading_count"})

    groups = [df_reduced]
    bins = df_reduced['time_utc'].tolist()
    return groups, bins


def get_processed_df(df, zone):
    
    cur_df = df[df['BZone'] == zone]
    groups, bins = get_time_resolution_groups(cur_df)
    columns = groups[0].columns.tolist()
    df_final = pd.DataFrame(columns=columns)
    for ind, group in enumerate(groups, 1):
#         print(f'#{ind}, shape: {group.shape}')
        df_final = pd.concat([df_final, group])
        
    df_final = df_final.sort_values('time_utc')

#     print()
#     print("final shape", df_final.shape)
    return df_final



# +
#Pipeline

for date in date_batches:

    start=time.time()
    print("Begin grouping data for date: {}".format(date))
    
    # Read in Methane Data
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder
    file_name=f'{date_batches[0]}_meth_weather_region.parquet.gzip'
    data_location = 's3://{}/{}'.format(s3_path, file_name)
    df = pd.read_parquet(data_location)
    df['time_utc'] = pd.to_datetime(df['time_utc']).dt.tz_localize(None)
    df['time_utc_hour'] = df['time_utc'].dt.round('h')

    ZONES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
           '13', '14', '15', '16']

    regions_combined = pd.DataFrame()
    
    for ZONE in ZONES:
        df_cur = get_processed_df(df, ZONE)      
        #combine each batch df into a single df and write to S3
        if regions_combined.shape[0] == 0:
            regions_combined = df_cur
        else:
            regions_combined = regions_combined.append(df_cur)
        
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder
    file_name=f'{date}_meth_weather_region_grouped.parquet.gzip'
    regions_combined.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')
    print("Completed grouping data for date in {:.2f} seconds".format(time.time()-start))
            
            

        
