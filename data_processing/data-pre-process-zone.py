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

# ### Goal
#
# Add:
# * distance calculation
# * reading count
# * qa_weightage
#
# Breakdown:
# * time units 
# * resolution 

# +
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import altair as alt


import argparse
import glob
import time
import sklearn
import numpy as np
import pandas as pd
import geojson
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
from descartes import PolygonPatch

alt.data_transformers.disable_max_rows()
# -

# ### Operations to apply
#
# * https://heartbeat.comet.ml/working-with-geospatial-data-in-machine-learning-ad4097c7228d
#
#
# * `dist_away` - from rounded points to the true point
# * `reading_count` - When aggregating over a given spot, number of readings we have for that spot
# * instead of every day, change data granularity to every week, or every 10 days etc. this would make missing data problem "dissappear"
# * For row (if day then also include week, if week then also include month granularity)
# * potentially instead of lat/lon we can do x,y,z... (may be unnecessary since we are just focussed on CA)



# ### Toggles
# * time - grouping any day increment `[ '1D', '3D', '5D', '7D', 10D']`
# #### Different Day Groupings
# * https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

# ### Load Data

# +
start = time.time()
s3_file_path = 's3://methane-capstone/data/combined-raw-data/combined-raw-facility-oil-weather.parquet.gzip'

df = pd.read_parquet(s3_file_path)
df['time_utc'] = pd.to_datetime(df['time_utc'])
df['year_month'] = df.year_month.astype(str)
print(df.shape)
print(df.dtypes)
end = time.time()
print("Load time", end-start)
# -

df.head()



#Helper function to calculate distance between raw lat/lon and rounded lat/lon
def haversine_distance(row, rounded_pair, unit = 'km'):

    lat_s, lon_s = row['lat'], row['lon'] #Source
    lat_d, lon_d = row[rounded_pair[0]], row[rounded_pair[1]] #Rounded Point - row['Destination Lat'], row['Destination Long']
    radius = 6371 if unit == 'km' else 3956 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.

    dlat = np.radians(lat_d - lat_s)
    dlon = np.radians(lon_d - lon_s)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat_s)) * np.cos(np.radians(lat_d)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c
    return distance



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

    print()
    print("final shape", df_final.shape)
    return df_final

bucket = 'methane-capstone'
subfolder = 'data/data-variants-zone'
s3_path = bucket+'/'+subfolder

def write_to_s3(dataframe, zone):

    file_name=f'data_{zone}.parquet.gzip'
    file_path = 's3://{}/{}'.format(s3_path, file_name)
    print(file_path)
    dataframe.to_parquet(file_path, compression='gzip')
    


# -

# ### Run Data Processing

df_cur.time_utc.unique()

df_cur.head()

df_cur = df_cur.set_index('time_utc').asfreq('D')

df_cur = df_cur.reset_index()

df_cur['time_utc'] = pd.to_datetime(df_cur['time_utc'])

df_cur[df_cur['time_utc'] == date]

len(df_cur)

len(df_cur[df_cur['reading_count'].isnull()]['time_utc'].tolist())

1038-81

# +
# Graph # of readings we have over time
time_unit = 'time_utc:T'

#Plot Interpolated vs. Raw
alt.Chart(df_cur, title="Reading Count - Zone 16").mark_point(point=True, tooltip=True).encode(
    x=alt.X(time_unit),
    y=alt.Y('reading_count:Q')
).properties(
    width=800,
    height=300
).interactive()
# -

reading_count_over_time



# +
ZONES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16']

for ZONE in ZONES:

    print("ZONE", ZONE)
    start = time.time()
    df_cur = get_processed_df(df, ZONE)
    write_to_s3(df_cur, ZONE)
    print("total time", time.time() - start)
        
# -







import subprocess
import re

s3_path

# ## Combine all the zone wise splits into one data set

zone_files = subprocess.check_output(['aws','s3','ls', f's3://{s3_path}/'])
zone_files = [entry.decode('utf-8') for entry in zone_files.split()]
r = re.compile("parquet.gzip")
zone_files = [x for x in zone_files if 'parquet' in x ]
zone_files

'data_15.parquet.gzip'.split("_")[1].split('.')[0]

[15]*10

# +
df_final = None

for ind, zone_file in enumerate(zone_files, 1):
    
    cur_zone = zone_file.split("_")[1].split('.')[0]
    cur_path = f's3://{s3_path}/{zone_file}'
    cur_df = pd.read_parquet(cur_path)    
    cur_df.insert(1, 'BZone', [int(cur_zone)]*len(cur_df))
    print(zone_file, cur_df.shape)
    
    if ind == 1:
        df_final = pd.DataFrame(columns = cur_df.columns)
    df_final = pd.concat([df_final, cur_df])
    
    
# -

df_final.shape

df_final.columns

file_name=f'data-zone-combined.parquet.gzip'
file_path = 's3://{}/{}'.format('methane-capstone/data/combined-raw-data', file_name)
print(file_path)
df_final.to_parquet(file_path, compression='gzip')

df_final.head()


