# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (TensorFlow 2.3 Python 3.7 CPU Optimized)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/tensorflow-2.3-cpu-py37-ubuntu18.04-v1
# ---

# ## Goal 
#
# * Load in all of our data, and select a specific region that we want to model over time.
# * Build out an auto-encoder anomaly detection model that models this region

# +
# # !pip install geopandas altair geojson matplotlib plotly descartes
# -

# ### Imports

# +
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import altair as alt


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
from datetime import date
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
from descartes import PolygonPatch
# -

#Altair puts a limit on plotting only 5000 rows from a pd.DataFrame. This line gets rid of that limit
alt.data_transformers.disable_max_rows()

# ### Load all data

# +
start = time.time()
s3_file_path = 's3://methane-capstone/data/combined-raw-data/combined-raw.parquet.gzip'

df = pd.read_parquet(s3_file_path)
df['time_utc'] = pd.to_datetime(df['time_utc'])
df['year_month'] = df.year_month.astype(str)
print(df.shape)
print(df.dtypes)
end = time.time()
print("Load time", end-start)
# -

# ### Check for missing days
#
# * This is across all data

# +
start = time.time()
dates_with_data = set(df['time_utc'].dt.strftime('%Y-%m-%d'))
all_dates = set(pd.period_range(datetime(2018,11,28), datetime(2021,9,30)).to_series().astype(str))
end = time.time()

print(end-start)
# -

#Dates we don't have ANY data for
missing_dates = sorted(list(all_dates - dates_with_data))
print("Missing days: ", len(missing_dates))
missing_dates

# ### Load in CA Base Map to Validate data on

# +
geo_json_path = "../data_processing/resources/california.geojson"
with open(geo_json_path) as json_file:
    geojson_data = geojson.load(json_file)
ca_poly = geojson_data['geometry']

gdf = gpd.read_file(geo_json_path)
choro_json = json.loads(gdf.to_json())
choro_data = alt.Data(values=choro_json['features'])

# Create Base CA Map
ca_base = alt.Chart(choro_data, title = 'California ').mark_geoshape(
    color='lightgrey',
    opacity=0.3,
    stroke='black',
    strokeWidth=1
).encode().properties(
    width=300,
    height=300
)
# -

# ## Geo Trimming
# * Either set your region
# * Or pick one exact point, and resolution
# * All readings will be averaged once per day regardles!
#
#
# For now, this is dodging the need to add spatial features to a model by restricting our data to a specific region
#
#

# +
# Geo Mask 
# 1. go to http://bboxfinder.com/
# 2. On the left side bar you'll find a option to draw a box
# 3. Copy/Paste the one liner of the box into the variables below

region = False
ll_lon, ll_lat, ur_lon, ur_lat = -123.215983,37.189673,-121.562541,38.236916 

#####  ~~ OR ~~ ##### 

resolution = .5       #can be --> [.1, .2, .5, 1.0]
rounded_lat, rounded_lon = 35.0, -119.0        #Needs to correspond to `resolution`

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

if region:
    
    print("REGION")
    geo_mask = (df.lon > ll_lon) & \
               (df.lat > ll_lat) & \
               (df.lon < ur_lon) & \
               (df.lat < ur_lat)
    trim_cols = ['time_utc','lat','lon', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision' 'methane_mixing_ratio_bias_corrected']

else:
    
    print("SPOT resolution", resolution)
    
    if resolution == 0.1:
        geo_mask = (df.rn_lat_1 == rounded_lat) & \
                   (df.rn_lon_1 == rounded_lon)
        trim_cols = ['time_utc','rn_lat_1','rn_lon_1', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']
        
    elif resolution == 0.2:
        geo_mask = (df.rn_lat_2 == rounded_lat) & \
                   (df.rn_lon_2 == rounded_lon)
        trim_cols = ['time_utc','rn_lat_2','rn_lon_2', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

    elif resolution == 0.5:
        geo_mask = (df.rn_lat_5 == rounded_lat) & \
                   (df.rn_lon_5 == rounded_lon)
        trim_cols = ['time_utc','rn_lat_5','rn_lon_5', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']
        
    elif resolution == 1.0:
        geo_mask = (df.rn_lat == rounded_lat) & \
                   (df.rn_lon == rounded_lon)
        trim_cols = ['time_utc','rn_lat','rn_lon', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']


df_trim = df[geo_mask][trim_cols]
print(f"Losing {'{:,}'.format(df.shape[0] - df_trim.shape[0])} rows")
print()
print(f"df_trim: {df_trim.shape}")
print()
df_trim.head()
# -

# ### Validate Data is in the region you specified

# +
if region:
    lat_str = 'lat'
    lon_str = 'lon'
    point_size = 20
else:
    if resolution == 0.1:
        lat_str = 'rn_lat_1'
        lon_str = 'rn_lon_1'
    elif resolution == 0.2:
        lat_str = 'rn_lat_2'
        lon_str = 'rn_lon_2'
    elif resolution == 0.5:
        lat_str = 'rn_lat_5'
        lon_str = 'rn_lon_5'
    elif resolution == 1.0:
        lat_str = 'rn_lat'
        lon_str = 'rn_lon'
    point_size = 100
    
    
#Plot all the readings
points = alt.Chart(df_trim).mark_circle(size=point_size).encode(
    longitude=f'{lon_str}:Q',
    latitude=f'{lat_str}:Q',
    tooltip= list(df_trim.columns)
)

ca_base + points
# -

# ## Pre-Process Data
# * one reading per day
# * understand missing days
#
#
# #### Interpolation Methods:
# * https://towardsdatascience.com/how-to-interpolate-time-series-data-in-apache-spark-and-python-pandas-part-1-pandas-cff54d76a2ea
# * https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
#
# #### Upsampling to Hourly Data:
# * https://kanoki.org/2020/04/14/resample-and-interpolate-time-series-data/
#
#
# Documentation:
# * https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
#
#
# * ffill - Forward Fill
# * bfill - Backward Fill
# * Interpolation
#
# I chose to stick with `time` interpolation!
#
#

# +
#Group by date. one average reading per day
df_trim_day = df_trim.groupby(df_trim.time_utc.dt.date).agg({'methane_mixing_ratio_bias_corrected': 'mean',
                                                             'methane_mixing_ratio': 'mean',
                                                             'methane_mixing_ratio_precision': 'mean',
                                                            }).reset_index()

#To also get the number of readings we have each day
df_trim_reading_count = df_trim.groupby(df_trim.time_utc.dt.date).agg({
                                                             'methane_mixing_ratio': 'count',
                                                            }).reset_index().rename(columns={"methane_mixing_ratio": "reading_count"})

#Join this onto our data set
df_trim_day = df_trim_day.merge(df_trim_reading_count, 'inner')

#Add in missing date rows
df_trim_day = df_trim_day.set_index('time_utc').asfreq('D')


#MISSING DAYS
miss_list = df_trim_day.isnull().sum(axis=0).tolist()
missing_dates = df_trim_day[df_trim_day['methane_mixing_ratio'].isnull()].index
print(f"Missing #: {miss_list[0]}")
print(f"Missing {100*miss_list[0]/df_trim_day.shape[0]}% of data")
print()

#Interpolate Data
method = 'time' #linear, spline, time
order = 3

df_interpol = df_trim_day.resample('D').mean()
df_trim_day['mmrbc_i'] = df_interpol['methane_mixing_ratio_bias_corrected'].interpolate(method=method) #, order=order)
df_trim_day['mmr_i'] = df_interpol['methane_mixing_ratio'].interpolate(method=method) #, order=order)
df_trim_day['mmrp_i'] = df_interpol['methane_mixing_ratio_precision'].interpolate(method=method) #, order=order)

#Set any interpolated date to `0`
df_trim_day['reading_count'] = df_trim_day['reading_count'].fillna(value=0)

df_trim_day = df_trim_day.reset_index()
print(df_trim_day.shape)
df_trim_day.head(4)
# -
# # START JACLYN WORK ********


# +
### THIS CELL WAS INCORRECTLY CALCULATING MISSING VALUES



# resolution = 1.0       #can be --> [.1, .2, .5, 1.0]
# min_lat = 32.0
# max_lat = 43.0
# min_lon = -124.0
# max_lon = -114.0
# rounded_lat = np.arange(min_lat, max_lat, resolution).tolist()
# rounded_lon = np.arange(min_lon, max_lon, resolution).tolist()

# ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
# ja_df = pd.DataFrame(columns = ['lat', 'lon', 'pct_missing', 'pct_missing_2018', 'pct_missing_2019', 'pct_missing_2020', 'pct_missing_2021'])
# # ja_df = pd.DataFrame(columns = ['lat', 'lon', 'pct_missing'])

# columns = list(ja_df)
# pct_missing_data = []


# for i in range(len(rounded_lat)):
#     for j in range(len(rounded_lon)):
    
# #         print("SPOT resolution", resolution)

#         if resolution == 0.1:
#             geo_mask = (df.rn_lat_1 == rounded_lat[i]) & \
#                        (df.rn_lon_1 == rounded_lon[j])
#             trim_cols = ['time_utc','rn_lat_1','rn_lon_1', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

#         elif resolution == 0.2:
#             geo_mask = (df.rn_lat_2 == rounded_lat[i]) & \
#                        (df.rn_lon_2 == rounded_lon[j])
#             trim_cols = ['time_utc','rn_lat_2','rn_lon_2', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

#         elif resolution == 0.5:
#             geo_mask = (df.rn_lat_5 == rounded_lat[i]) & \
#                        (df.rn_lon_5 == rounded_lon[j])
#             trim_cols = ['time_utc','rn_lat_5','rn_lon_5', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

#         elif resolution == 1.0:
#             geo_mask = (df.rn_lat == rounded_lat[i]) & \
#                        (df.rn_lon == rounded_lon[j])
#             trim_cols = ['time_utc','rn_lat','rn_lon', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']


#         df_trim = df[geo_mask][trim_cols]
# #         print(f"Losing {'{:,}'.format(df.shape[0] - df_trim.shape[0])} rows")
# #         print()
# #         print(f"df_trim: {df_trim.shape}")
#         if len(df_trim) == 0:
#             continue
# #         print()
# #         df_trim.head()


#         #Group by date. one average reading per day
#         df_trim_day = df_trim.groupby(df_trim.time_utc.dt.date).agg({'methane_mixing_ratio_bias_corrected': 'mean',
#                                                                      'methane_mixing_ratio': 'mean',
#                                                                      'methane_mixing_ratio_precision': 'mean',
#                                                                     }).reset_index()

#         #To also get the number of readings we have each day
#         df_trim_reading_count = df_trim.groupby(df_trim.time_utc.dt.date).agg({
#                                                                      'methane_mixing_ratio': 'count',
#                                                                     }).reset_index().rename(columns={"methane_mixing_ratio": "reading_count"})

#         #Join this onto our data set
#         df_trim_day = df_trim_day.merge(df_trim_reading_count, 'inner')

#         #Add in missing date rows
#         df_trim_day = df_trim_day.set_index('time_utc').asfreq('D')


#         #MISSING DAYS
#         miss_list = df_trim_day.isnull().sum(axis=0).tolist()
#         missing_dates = df_trim_day[df_trim_day['methane_mixing_ratio'].isnull()].index
# #         print(f"Missing #: {miss_list[0]}")
# #         print(f"Missing {100*miss_list[0]/df_trim_day.shape[0]:.2f}% of data")
#         pct_missing = 100*miss_list[0]/df_trim_day.shape[0]
#         try:
#             pct_missing_2018 = 100*len(missing_dates[missing_dates.year ==2018])/len(df_trim_day[df_trim_day.index.year == 2018])
#         except:
#             pct_missing_2018 = 1
#         try:
#             pct_missing_2019 = 100*len(missing_dates[missing_dates.year ==2019])/len(df_trim_day[df_trim_day.index.year == 2019])
#         except:
#             pct_missing_2019 = 1
#         try:
#             pct_missing_2020 = 100*len(missing_dates[missing_dates.year ==2020])/len(df_trim_day[df_trim_day.index.year == 2020])
#         except:
#             pct_missing_2020 = 1
#         try:
#             pct_missing_2021 = 100*len(missing_dates[missing_dates.year ==2021])/len(df_trim_day[df_trim_day.index.year == 2021])
#         except:
#             pct_missing_2021 = 1
# #         print(f"{pct_missing:.2f}")
# #         print()
#         zipped = zip(columns, [rounded_lat[i], rounded_lon[j], pct_missing, pct_missing_2018, pct_missing_2019, pct_missing_2020, pct_missing_2021])
# #         zipped = zip(columns, [rounded_lat[i], rounded_lon[j], pct_missing])
#         pct_missing_dict = dict(zipped)
#         pct_missing_data.append(pct_missing_dict)


# ja_df = ja_df.append(pct_missing_data, True)

# print(ja_df)

# +
# USE THIS CELL TO AGGREGATE ON ONE DAY ONLY

resolution = 0.1       #can be --> [.1, .2, .5, 1.0]
min_lat = 32.0
max_lat = 43.0
min_lon = -124.0
max_lon = -114.0
rounded_lat = np.arange(min_lat, max_lat, resolution).tolist()
rounded_lon = np.arange(min_lon, max_lon, resolution).tolist()

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
ja_df = pd.DataFrame(columns = ['lat', 'lon', 'pct_missing', 'pct_missing_2018', 
                                'pct_missing_2019', 'pct_missing_2020', 'pct_missing_2021'])
# ja_df = pd.DataFrame(columns = ['lat', 'lon', 'pct_missing'])

columns = list(ja_df)
pct_missing_data = []


for i in range(len(rounded_lat)):
    for j in range(len(rounded_lon)):
    
#         print("SPOT resolution", resolution)

        if resolution == 0.1:
            geo_mask = (df.rn_lat_1 == rounded_lat[i]) & \
                       (df.rn_lon_1 == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat_1','rn_lon_1', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

        elif resolution == 0.2:
            geo_mask = (df.rn_lat_2 == rounded_lat[i]) & \
                       (df.rn_lon_2 == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat_2','rn_lon_2', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

        elif resolution == 0.5:
            geo_mask = (df.rn_lat_5 == rounded_lat[i]) & \
                       (df.rn_lon_5 == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat_5','rn_lon_5', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

        elif resolution == 1.0:
            geo_mask = (df.rn_lat == rounded_lat[i]) & \
                       (df.rn_lon == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat','rn_lon', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']


        df_trim = df[geo_mask][trim_cols]
#         print(f"Losing {'{:,}'.format(df.shape[0] - df_trim.shape[0])} rows")
#         print()
#         print(f"df_trim: {df_trim.shape}")
        if len(df_trim) == 0:
            continue
#         print()
#         df_trim.head()


        #Group by date. one average reading per day
        df_trim_day = df_trim.groupby(df_trim.time_utc.dt.date).agg({'methane_mixing_ratio_bias_corrected': 'mean',
                                                                     'methane_mixing_ratio': 'mean',
                                                                     'methane_mixing_ratio_precision': 'mean',
                                                                    }).reset_index()
        df_trim_day['time_utc'] = pd.to_datetime(df_trim_day['time_utc'], utc=True)
        df_trim_day = df_trim_day.set_index('time_utc')

        pct_missing = 100*(1-df_trim_day.shape[0]/1038)
        try:
            pct_missing_2018 = 100*(1-len(df_trim_day[df_trim_day.index.year == 2018])/34)
        except:
            pct_missing_2018 = 1
        try:
            pct_missing_2019 = 100*(1-len(df_trim_day[df_trim_day.index.year == 2019])/365)
        except:
            pct_missing_2019 = 1
        try:
            pct_missing_2020 = 100*(1-len(df_trim_day[df_trim_day.index.year == 2020])/366)
        except:
            pct_missing_2020 = 1
        try:
            pct_missing_2021 = 100*(1-len(df_trim_day[df_trim_day.index.year == 2021])/273)
        except:
            pct_missing_2021 = 1
#         print(f"{pct_missing:.2f}")
#         print()
        zipped = zip(columns, [rounded_lat[i], rounded_lon[j], pct_missing, pct_missing_2018, 
                               pct_missing_2019, pct_missing_2020, pct_missing_2021])
#         zipped = zip(columns, [rounded_lat[i], rounded_lon[j], pct_missing])
        pct_missing_dict = dict(zipped)
        pct_missing_data.append(pct_missing_dict)


ja_df = ja_df.append(pct_missing_data, True)

# print(ja_df)

# +
# USE THIS CELL TO CREATE 7 DAY AGGREGATES

start_day = date(2018,11, 28)

resolution = 1.0       #can be --> [.1, .2, .5, 1.0]
min_lat = 32.0
max_lat = 43.0
min_lon = -124.0
max_lon = -114.0
rounded_lat = np.arange(min_lat, max_lat, resolution).tolist()
rounded_lon = np.arange(min_lon, max_lon, resolution).tolist()

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
ja_df = pd.DataFrame(columns = ['lat', 'lon', 'pct_missing', 'pct_missing_2018', 
                                'pct_missing_2019', 'pct_missing_2020', 'pct_missing_2021'])


columns = list(ja_df)
pct_missing_data = []


for i in range(len(rounded_lat)):
    for j in range(len(rounded_lon)):
    

        if resolution == 0.1:
            geo_mask = (df.rn_lat_1 == rounded_lat[i]) & \
                       (df.rn_lon_1 == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat_1','rn_lon_1', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

        elif resolution == 0.2:
            geo_mask = (df.rn_lat_2 == rounded_lat[i]) & \
                       (df.rn_lon_2 == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat_2','rn_lon_2', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

        elif resolution == 0.5:
            geo_mask = (df.rn_lat_5 == rounded_lat[i]) & \
                       (df.rn_lon_5 == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat_5','rn_lon_5', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

        elif resolution == 1.0:
            geo_mask = (df.rn_lat == rounded_lat[i]) & \
                       (df.rn_lon == rounded_lon[j])
            trim_cols = ['time_utc','rn_lat','rn_lon', 'qa_val', 
                         'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']


        df_trim = df[geo_mask][trim_cols]

        if len(df_trim) == 0:
            continue

            
        df_trim['time_utc'] = pd.to_datetime(df_trim['time_utc'], utc=True)
        df_trim_day = df_trim.set_index('time_utc').groupby(pd.Grouper(freq="7D", origin="start_day")).agg({'methane_mixing_ratio_bias_corrected': 'mean',
                                                                     'methane_mixing_ratio': 'mean',
                                                                     'methane_mixing_ratio_precision': 'mean',
                                                                    })

        df_trim_notnull = df_trim_day[df_trim_day['methane_mixing_ratio_bias_corrected'].notnull()]



        pct_missing = 100*(1-df_trim_notnull.shape[0]/148)
        try:
            pct_missing_2018 = 100*(1-len(df_trim_notnull[df_trim_notnull.index.year == 2018])/5)
        except:
            pct_missing_2018 = 1
        try:
            pct_missing_2019 = 100*(1-len(df_trim_notnull[df_trim_notnull.index.year == 2019])/52)
        except:
            pct_missing_2019 = 1
        try:
            pct_missing_2020 = 100*(1-len(df_trim_notnull[df_trim_notnull.index.year == 2020])/52)
        except:
            pct_missing_2020 = 1
        try:
            pct_missing_2021 = 100*(1-len(df_trim_notnull[df_trim_notnull.index.year == 2021])/39)
        except:
            pct_missing_2021 = 1

        zipped = zip(columns, [rounded_lat[i], rounded_lon[j], pct_missing, pct_missing_2018, 
                               pct_missing_2019, pct_missing_2020, pct_missing_2021])

        pct_missing_dict = dict(zipped)
        pct_missing_data.append(pct_missing_dict)


ja_df = ja_df.append(pct_missing_data, True)

# print(ja_df)

# +
# Create Base CA Map
ca_base = alt.Chart(choro_data, title = '1.0 Resolution').mark_geoshape(
    color='lightgrey',
    opacity=0.3,
    stroke='black',
    strokeWidth=1
).encode().properties(
    width=300,
    height=300
)


scale = alt.Scale(
    domain=[100, 50, 0],
    range=['darkred', 'orange', 'green'],
    type='linear'
)
    
#Plot all the readings
points = alt.Chart(ja_df).mark_circle(size=100).encode(
    longitude='lon',
    latitude='lat',
    tooltip= list(ja_df.columns),
#     size = 'pct_missing',
    color=alt.Color('pct_missing_2021', scale=scale)
)





ca_base + points

# +
pct_missing_hist = alt.Chart(ja_df, title="Percent Missing Data").mark_bar(tooltip=True).encode(
    alt.X("pct_missing:Q", bin=True),
    y='count()',
).interactive()

pct_missing_hist
# -

# # END JACLYN WORK ********** ************* ********* *******

# ### Understand Missing Data
# * Do this via the reading count

df_trim_day.groupby(df_trim_day.time_utc.dt.year)['reading_count'].sum()

# +
# Graph # of readings we have over time
time_unit = 'yearmonth(time_utc):T'

#Plot Interpolated vs. Raw
reading_count_over_time = alt.Chart(df_trim_day, title="Reading Count Over Time").mark_line(point=True, tooltip=True).encode(
    x=alt.X(time_unit),
    y=alt.Y('mean(reading_count):Q')
).interactive()


reading_count_hist = alt.Chart(df_trim_day, title="Reading Count Histogram").mark_bar(tooltip=True).encode(
    alt.X("reading_count:Q", bin=True),
    y='count()',
).interactive()

reading_count_over_time | reading_count_hist

# +
time_unit = 'time_utc:T'
point_size = 10

#Plot Interpolated vs. Raw
non_interpo = alt.Chart(df_trim_day).mark_line().encode(
    x=alt.X(time_unit),
    y=alt.Y('methane_mixing_ratio_bias_corrected:Q'),
    color=alt.value("#FFAA00")
).interactive()

interpo = alt.Chart(df_trim_day).mark_line().encode(
    x=alt.X(time_unit),
    y=alt.Y('mmrbc_i:Q'),
    color=alt.value("#0000FF")
).interactive()


print("Blue: Interpolated")
print("Yellow: Non-Interpolated")
non_interpo | interpo

# +
# time_unit = 'time_utc:T'
# point_size = 10

# #Plot Interpolated vs. Raw
# non_interpo = alt.Chart(df_trim_day).mark_line().encode(
#     x=alt.X(time_unit),
#     y=alt.Y('methane_mixing_ratio_bias_corrected:Q'),
#     color=alt.value("#FFAA00")
# ).interactive()


# interpo = alt.Chart(df_trim_day).mark_line().encode(
#     x=alt.X(time_unit),
#     y=alt.Y('mmrbc_i:Q'),
#     color=alt.value("#0000FF")
# ).interactive()


# print("Blue: Interpolated")
# print("Yellow: Non-Interpolated")
# non_interpo | interpo
# -

# ## Auto-encoder with LSTM 
#
# * Focussing on just `mmrbc_i`
#
# Autoencoder is a neural network framework that tries to reconstruct the data that we are passing it. It will take the incoming data, and project it down to a latent vector space that is smaller than the shape fed in. Then it will decode from this latent vector space back to the input dimension hoping to minimize the error betwee input and output.
#
# We are trying to get really good at the data that we are given, so that if something doesn't look like that data, we will be able to classify this as an anomaly

# +
# # !pip install tensorflow keras seaborn
# # !pip install tensorflow==2.5.0

# +
import tensorflow as tf
from tensorflow import keras
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 18

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# +
print(df_trim_day.shape)
print("start_dt:", df_trim_day['time_utc'].min(), "\nend_dt:", df_trim_day['time_utc'].max(), "\nnumber_days:", df_trim_day['time_utc'].max() - df_trim_day['time_utc'].min())
print()

df_trim_day = df_trim_day.set_index('time_utc')

df_trim_day.head()
# -

plt.plot(df_trim_day['mmrbc_i'], label='mmrbc_i')

# ### Train/Test Split 
# * by date 
# * by data size

# +
by_dt = True
date_threshold = '2021-01-01'
train_percent = 0.95

if by_dt:
    train, test = df_trim_day.loc[df_trim_day.index < date_threshold], df_trim_day.loc[df_trim_day.index >= date_threshold]
        
else:
    
    train_size = int(len(df_trim_day) * train_percent)
    test_size = len(df_trim_day) - train_size
    train, test = df_trim_day.iloc[:train_size], df_trim_day.iloc[train_size:]
    

print(train.shape, test.shape)
# -

# ### Scale Values
# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset

scaler = StandardScaler()
scaler = scaler.fit(train[['mmrbc_i']])


# +
train['mmrbc_i'] = scaler.transform(train[['mmrbc_i']])
test['mmrbc_i'] = scaler.transform(test[['mmrbc_i']])

train.head()
# -

# ### Create Windows for LSTM
#
# As required for LSTM networks, we require to reshape an input data into `n_samples` x `timesteps` x `n_features`. 
#
# * Because our data is by day, I will say a sequence is **1 week** worth of time

# +
# Number of time steps to look back. Larger sequences (look further back) may improve forecasting.
seq_size = 7

def create_sequenes(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        #print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)


trainX, trainY = create_sequenes(train[['mmrbc_i']], train['mmrbc_i'], seq_size)
testX, testY = create_sequenes(test[['mmrbc_i']], test['mmrbc_i'], seq_size)


print("trainX.shape: ", trainX.shape, "trainY.shape: ", trainY.shape)
print("testX.shape: ", testX.shape, "testY.shape: ", testY.shape)
# -

# ### Baseline Model
#
# * this is where we have the freedom to play with our model architecture

# +
model = keras.Sequential()

#Here we can include other things like readings per day.
model.add(keras.layers.LSTM(units=64, input_shape = (trainX.shape[1], trainX.shape[2])))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=trainX.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense( units=trainX.shape[2])))
model.compile(loss='mae', optimizer='adam')
model.summary()
# -

# ### Start Training

# +

history = model.fit(
    trainX, trainY,
    epochs = 10,
    batch_size=32,
    validation_split = 0.1,
    shuffle= False #No shuffle cause time series!
)

# -

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

# ### Predict on Train

trainX_pred = model.predict(trainX)
train_mae_loss = np.mean(np.abs(trainX_pred, trainX), axis=1)
sns.distplot(train_mae_loss, bins=50, kde=True)    

# ### Predict on Test

# +
testX_pred = model.predict(testX)
test_mae_loss = np.mean(np.abs(testX_pred, testX), axis=1)

# Try not to look at this plot haha! (or just do it cuz why not)
# sns.distplot(test_mae_loss, bins=50, kde=True)    
# -

# ### Selecting the threshold is key!
#
# * here we can look at the training loss's and let that guid our choice for the threshold
# * lower the value, more anomalies we will detect.
# * We can reason about this with the descriptive statistics we have learned from the data

anom_thresh = 1.5

# +

test_score_df = pd.DataFrame(index=test[seq_size:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = anom_thresh
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['mmrbc_i'] = test[seq_size:].mmrbc_i

# -

plt.plot(test_score_df.index, test_score_df.loss, label = 'loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Test Loss vs. Anomaly Loss Threshold")
plt.legend()

anomalies = test_score_df[test_score_df.anomaly]

# +
plt.plot(
    test[seq_size:].index,
    scaler.inverse_transform(test[seq_size:].mmrbc_i),
    label = 'mmrbc_i'
)

sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.mmrbc_i),
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
# -


