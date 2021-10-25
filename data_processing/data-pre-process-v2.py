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

# # Goal
#
# Combine Weather-Methane data with CA-Vista Data

# +
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# import altair as alt

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

# alt.data_transformers.disable_max_rows()
# -

# # Load Data

#IMPORT METHANE DATA
data_location='s3://methane-capstone/data/weather/era5/ALL_methane_and_weather.parquet'
methane_weather_df = pd.read_parquet(data_location)
methane_weather_df['time_utc_hour'] = methane_weather_df['time_utc'].dt.round('h')
print(methane_weather_df.shape)
methane_weather_df.head(2)

#IMPORT FACILITY DATA
data_location='s3://methane-capstone/data/combined-raw-data/combined-raw-facility-details.parquet.gzip'
facility_df = pd.read_parquet(data_location)
facility_df['time_utc_hour'] = facility_df['time_utc'].dt.round('h')
print(facility_df.shape)
facility_df.head(2)

# +
#IMPORT OIL WELL DATA
data_location='s3://methane-capstone/data/oil-well-data/og_wells_rn_1.parquet.gzip'
og_wells_rn_1_df = pd.read_parquet(data_location)
og_wells_rn_1_df['well_count_rn_1'] = og_wells_rn_1_df['well_count']

data_location='s3://methane-capstone/data/oil-well-data/og_wells_rn_2.parquet.gzip'
og_wells_rn_2_df = pd.read_parquet(data_location)
og_wells_rn_2_df['well_count_rn_2'] = og_wells_rn_2_df['well_count']

data_location='s3://methane-capstone/data/oil-well-data/og_wells_rn_5.parquet.gzip'
og_wells_rn_5_df = pd.read_parquet(data_location)
og_wells_rn_5_df['well_count_rn_5'] = og_wells_rn_5_df['well_count']

data_location='s3://methane-capstone/data/oil-well-data/og_wells_rn.parquet.gzip'
og_wells_rn_df = pd.read_parquet(data_location)
og_wells_rn_df['well_count_rn'] = og_wells_rn_df['well_count']

print(og_wells_rn_1_df.shape)
print(og_wells_rn_2_df.shape)
print(og_wells_rn_5_df.shape)
print(og_wells_rn_df.shape)

# og_wells_rn_1_df.head(2)
# og_wells_rn_2_df.head(2)
# og_wells_rn_5_df.head(2)
og_wells_rn_df.head(2)
# -

# # Merge Data


# #### Merge weather_methane with weather_facilties

# +
print(methane_weather_df.shape)
print(methane_weather_df.columns)

print(facility_df.shape)
print(facility_df.columns)
# -

methane_weather__facility_df = pd.merge(methane_weather_df, facility_df[['time_utc_hour', 'lat', 'lon','point_type', 'inFacility']], how='inner', left_on = ['time_utc_hour', 'lat', 'lon'], right_on = ['time_utc_hour','lat','lon'])
print(methane_weather__facility_df.shape)
methane_weather__facility_df.head(5)

# #### Merge weather_methane_facilties with oil wells
#

print(methane_weather__facility_df.shape)
print(methane_weather__facility_df.columns)
print("########################################################################################")
print(og_wells_rn_1_df.shape)
print(og_wells_rn_1_df.columns)
print("########################################################################################")
print(og_wells_rn_2_df.shape)
print(og_wells_rn_2_df.columns)
print("########################################################################################")
print(og_wells_rn_5_df.shape)
print(og_wells_rn_5_df.columns)
print("########################################################################################")
print(og_wells_rn_df.shape)
print(og_wells_rn_df.columns)

# og_wells_rn_1
methane_weather__facility_oil_df = pd.merge(methane_weather__facility_df, og_wells_rn_1_df[['rn_lat_1', 'rn_lon_1', 'well_count_rn_1']], how='left', left_on = ['rn_lat_1', 'rn_lon_1'], right_on = ['rn_lat_1', 'rn_lon_1'])
print(methane_weather__facility_oil_df.shape)
methane_weather__facility_oil_df.head(2)

# og_wells_rn_2
methane_weather__facility_oil_df = pd.merge(methane_weather__facility_oil_df, og_wells_rn_2_df[['rn_lat_2', 'rn_lon_2', 'well_count_rn_2']], how='left', left_on = ['rn_lat_2', 'rn_lon_2'], right_on = ['rn_lat_2', 'rn_lon_2'])
print(methane_weather__facility_oil_df.shape)
methane_weather__facility_oil_df.head(2)

# og_wells_rn_5
methane_weather__facility_oil_df = pd.merge(methane_weather__facility_oil_df, og_wells_rn_5_df[['rn_lat_5', 'rn_lon_5', 'well_count_rn_5']], how='left', left_on = ['rn_lat_5', 'rn_lon_5'], right_on = ['rn_lat_5', 'rn_lon_5'])
print(methane_weather__facility_oil_df.shape)
methane_weather__facility_oil_df.head(2)

# og_wells_rn
methane_weather__facility_oil_df = pd.merge(methane_weather__facility_oil_df, og_wells_rn_df[['rn_lat', 'rn_lon', 'well_count_rn']], how='left', left_on = ['rn_lat', 'rn_lon'], right_on = ['rn_lat', 'rn_lon'])
print(methane_weather__facility_oil_df.shape)
methane_weather__facility_oil_df.head(2)

# # EDA

print(methane_weather__facility_oil_df.shape)
print(methane_weather__facility_oil_df.columns)
methane_weather__facility_oil_df.head(5)

# +
#check point_type
print("original facility dataset)")
print(facility_df['point_type'].count())
print(facility_df['point_type'].unique())
print("###################################")

print("new combined dataset")
print(methane_weather__facility_oil_df['point_type'].count())
print(methane_weather__facility_oil_df['point_type'].unique())

#SHOULD MATCH!!!
# -

#check inFacility
print(facility_df['inFacility'].sum())
print(methane_weather__facility_oil_df['inFacility'].sum())

#check og_well_rn1
plt.scatter(methane_weather__facility_oil_df['rn_lon_1'],
            methane_weather__facility_oil_df['rn_lat_1'], c="yellow", alpha=0.1, marker='.')
plt.scatter(og_wells_rn_1_df['rn_lon_1'],og_wells_rn_1_df['rn_lat_1'], c="red", alpha=0.5, marker='.')
plt.scatter(methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_1>0]['rn_lon_1'],
            methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_1>0]['rn_lat_1'], c="blue", alpha=0.5, marker='.')

# +
#og_wells_rn_1_df.rn_lat_1.unique()
a=[32.5, 32.6, 32.7, 32.8, 32.9, 33.0, 33.1, 33.2, 33.3, 33.4, 33.5, 33.6,
   33.7, 33.8, 33.9, 34.0, 34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 
   34.9, 35.0, 35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9, 36.0, 
   36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7, 36.8, 36.9, 37.0, 37.1, 37.2, 
   37.3, 37.4, 37.5, 37.6, 37.7, 37.8, 37.9, 38.0, 38.1, 38.2, 38.3, 38.4, 
   38.5, 38.6, 38.7, 38.8, 38.9, 39.0, 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 
   39.7, 39.8, 39.9, 40.0, 40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7, 40.8, 
   40.9, 41.0, 41.1, 41.2, 41.3, 41.4, 41.5, 41.6, 41.7, 41.8, 41.9, 42.0]

#methane_weather__facility_oil_df.rn_lat_1.unique()
b=[32.5, 32.6, 32.7, 32.8, 32.9, 33.0, 33.1, 33.2, 33.3, 33.4, 33.5, 33.6, 
   33.7, 33.8, 33.9, 34.0, 34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 
   34.9, 35.0, 35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9, 36.0, 
   36.1, 36.2, 36.3, 36.4, 36.5, 36.6, 36.7, 36.8, 36.9, 37.0, 37.1, 37.2, 
   37.3, 37.4, 37.5, 37.6, 37.7, 37.8, 37.9, 38.0, 38.1, 38.2, 38.3, 38.4, 
   38.5, 38.6, 38.7, 38.8, 38.9, 39.0, 39.1, 39.2, 39.3, 39.4, 39.5, 39.6, 
   39.7, 39.8, 39.9, 40.0, 40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7, 40.8, 
   40.9, 41.0, 41.5, 41.8]
# -

#check og_well_rn2
plt.scatter(methane_weather__facility_oil_df['rn_lon_2'],
            methane_weather__facility_oil_df['rn_lat_2'], c="yellow", alpha=0.1, marker='.')
plt.scatter(og_wells_rn_2_df['rn_lon_2'],og_wells_rn_2_df['rn_lat_2'], c="red", alpha=0.5, marker='.')
plt.scatter(methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_2>0]['rn_lon_2'],
            methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_2>0]['rn_lat_2'], c="blue", alpha=0.5, marker='.')

#check og_well_rn5
plt.scatter(methane_weather__facility_oil_df['rn_lon_5'],
            methane_weather__facility_oil_df['rn_lat_5'], c="yellow", alpha=0.1, marker='.')
plt.scatter(og_wells_rn_5_df['rn_lon_5'],og_wells_rn_5_df['rn_lat_5'], c="red", alpha=0.5, marker='.')
plt.scatter(methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_5>0]['rn_lon_5'],
            methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_5>0]['rn_lat_5'], c="blue", alpha=0.5, marker='.')

#check og_well_rn
plt.scatter(methane_weather__facility_oil_df['rn_lon'],
            methane_weather__facility_oil_df['rn_lat'], c="yellow", alpha=0.1, marker='.')
plt.scatter(og_wells_rn_df['rn_lon'],og_wells_rn_df['rn_lat'], c="red", alpha=0.5, marker='.')
plt.scatter(methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_5>0]['rn_lon'],
            methane_weather__facility_oil_df[methane_weather__facility_oil_df.well_count_rn_5>0]['rn_lat'], c="blue", alpha=0.5, marker='.')

# ### Check Data Distribution

df_sample = methane_weather__facility_oil_df.sample(frac=0.005)
print(df_sample.shape)
df_sample.head()

boom = df_sample['methane_mixing_ratio_bias_corrected'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Methane Mixing Ratio - CA')
plt.xlabel('PPB')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)

methane_weather__facility_oil_df.isnull().sum()

# # Write to S3


gzip_write_path = 's3://methane-capstone/data/combined-raw-data/combined-raw-facility-oil-weather.parquet.gzip'
methane_weather__facility_oil_df.to_parquet(gzip_write_path, compression='gzip')
