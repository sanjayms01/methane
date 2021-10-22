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

# # EDA: ERA5 & METHANE MERGED DATA 

# +
# Initialize notebook environment.
# %matplotlib inline
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
import xarray as xr
import time

pd.set_option('display.float_format', lambda x: '%.5f' % x)  #I do not want to see scientific notations, e.g. 1.5e10
# -

# ## Imports

# +
# COMBINING MONTHLY WEATHER & METHANE DATA AND WRITING IT TO S3

# #IMPORT WEATHER & METHANE DATA

# available_data = ['2021.09', '2021.08',  '2021.07',  '2021.06', '2021.05', '2021.04', '2021.03',  '2021.02',  '2021.01',
# '2020.12',  '2020.11',  '2020.10',  '2020.09',  '2020.08',  '2020.07', '2020.06',  '2020.05',  '2020.04',  '2020.03',  '2020.02',  '2020.01',
# '2019.12',  '2019.11',  '2019.10',  '2019.09',  '2019.08',  '2019.07', '2019.06',  '2019.05',  '2019.04',  '2019.03',  '2019.02',  '2019.01',
# '2018.12',  '2018.11']

# methane_weather_df = pd.DataFrame()

# for year_month in available_data:
    
#     year=year_month.split(".")[0]
#     month=year_month.split(".")[1]

#     bucket = 'methane-capstone'
#     subfolder = 'data/weather/era5'
#     s3_path_month = bucket+'/'+subfolder
#     file_name='{}-{}_methane_and_weather.parquet'.format(year,month)
#     data_location = 's3://{}/{}'.format(s3_path_month, file_name)
#     weather_methane = pd.read_parquet(data_location)
    
#     methane_weather_df = pd.concat([weather_methane, methane_weather_df])

# #write WEATHER & METHANE data
# bucket = 'methane-capstone'
# subfolder = 'data/weather/era5'
# s3_path_month = bucket+'/'+subfolder
# file_name='ALL_methane_and_weather.parquet'
# data_location = 's3://{}/{}'.format(s3_path_month, file_name)
# methane_weather_df.to_parquet(data_location)
# methane_weather_df.head(2)

# +
#IMPORT METHANE DATA WITHOUT WEATHER

# Read in Methane Dataframe
bucket = 'methane-capstone'
subfolder = 'data/combined-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='combined-raw.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
methane_df = pd.read_parquet(data_location)

#IMPORT METHANE DATA WITH WEATHER

# Read in Methane Dataframe
bucket = 'methane-capstone'
subfolder = 'data/weather/era5'
s3_path_month = bucket+'/'+subfolder
file_name='ALL_methane_and_weather.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
methane_weather_df = pd.read_parquet(data_location)
# -

#methane data without weather
print(methane_df.shape)
methane_df.head(2)

#methane data with weather
print(methane_weather_df.shape)
methane_weather_df.head(2)

# ## Similarities

# +
# CHECK IF methane data and methane&weather data matches:

#Do # of rows match?
print(len(methane_df.shape) == len(methane_weather_df.shape))

# print(methane_df.columns)

#Do the dataframes with similar columns match?
methane_df.reset_index() == methane_weather_df[['time_utc', 'year_month', 'lat', 'lon', 'rn_lat_1', 'rn_lon_1',
       'rn_lat_2', 'rn_lon_2', 'rn_lat_5', 'rn_lon_5', 'rn_lat', 'rn_lon',
       'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision',
       'methane_mixing_ratio_bias_corrected']].reset_index() 

# +
#INSPECT WHERE THE TWO DATAFRAMES DO NOT MATCH
methane_df[methane_df.reset_index() != methane_weather_df[['time_utc', 'year_month', 'lat', 'lon', 'rn_lat_1', 'rn_lon_1',
       'rn_lat_2', 'rn_lon_2', 'rn_lat_5', 'rn_lon_5', 'rn_lat', 'rn_lon',
       'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision',
       'methane_mixing_ratio_bias_corrected']].reset_index() ]   

#John: The reason why some of the time_utc doesnt match is because I had to save part of the weather data timestamp in milliseconds 
#I had an error writing to parquet when the timestamp was in nanoseconds.  
#https://stackoverflow.com/questions/59682833/pyarrow-lib-arrowinvalid-casting-from-timestampns-to-timestampms-would-los
# -

# # Check Nulls

#CHECK NULLS
methane_weather_df.isnull().sum()

methane_weather_df.columns

# +
# Nulls have similar count, are they the same rows?

#First, reset index to show unique row numbers.
methane_weather_df = methane_weather_df.reset_index()
methane_weather_df.drop(['index'], axis=1,inplace=True)
methane_weather_df
# -

#Compare several variables
# e.g. are the nulls in air temp the same rows as the nulls in air pressure?
print(sum(methane_weather_df.index[methane_weather_df['air_pressure_at_mean_sea_level'].isnull()] == methane_weather_df.index[methane_weather_df['integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation'].isnull()]))
print(sum(methane_weather_df.index[methane_weather_df['air_temperature_at_2_metres'].isnull()] == methane_weather_df.index[methane_weather_df['lwe_thickness_of_surface_snow_amount'].isnull()]))
print(sum(methane_weather_df.index[methane_weather_df['air_temperature_at_2_metres_1hour_Maximum'].isnull()] == methane_weather_df.index[methane_weather_df['northward_wind_at_100_metres'].isnull()]))
print(sum(methane_weather_df.index[methane_weather_df['air_temperature_at_2_metres_1hour_Minimum'].isnull()] == methane_weather_df.index[methane_weather_df['northward_wind_at_10_metres'].isnull()]))
print(sum(methane_weather_df.index[methane_weather_df['dew_point_temperature_at_2_metres'].isnull()] == methane_weather_df.index[methane_weather_df['precipitation_amount_1hour_Accumulation'].isnull()]))
print(sum(methane_weather_df.index[methane_weather_df['eastward_wind_at_100_metres'].isnull()] == methane_weather_df.index[methane_weather_df['snow_density'].isnull()]))
print(sum(methane_weather_df.index[methane_weather_df['eastward_wind_at_10_metres'].isnull()] == methane_weather_df.index[methane_weather_df['surface_air_pressure'].isnull()]))

#looks to be the same rows.. Let's inspect these rows.
null_rows = methane_weather_df[['time_utc', 'year_month', 'rn_lat_2', 'rn_lon_2']][methane_weather_df['air_pressure_at_mean_sea_level'].isnull()]
null_rows

# +
#Print out the unique count of each groupby
print(null_rows.groupby('year_month').count().shape)   #35 unique year_month have nulls
print(null_rows.groupby('time_utc').count().shape)     #50688 unique utc time have nulls
print(null_rows.groupby('rn_lat_2').count().shape)     
print(null_rows.groupby('rn_lon_2').count().shape)
print(null_rows.groupby(['rn_lat_2','rn_lon_2']).count().shape)  #162 unique lat/lon combos have nulls

print("################################")
#For comparison
print(methane_df.groupby('year_month').count().shape)
print(methane_df.groupby('time_utc').count().shape)
print(methane_df.groupby('rn_lat_2').count().shape)
print(methane_df.groupby('rn_lon_2').count().shape)
print(methane_df.groupby(['rn_lat_2','rn_lon_2']).count().shape)
# -

#looks like about <10% of data have nulls for weather data
null_rows.groupby('year_month').count()

methane_df.groupby('year_month').count()

# ## General EDA

methane_weather_df[['air_pressure_at_mean_sea_level', 'air_temperature_at_2_metres',
       'air_temperature_at_2_metres_1hour_Maximum',
       'air_temperature_at_2_metres_1hour_Minimum',
       'dew_point_temperature_at_2_metres', 'eastward_wind_at_100_metres',
       'eastward_wind_at_10_metres',
       'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
       'lwe_thickness_of_surface_snow_amount', 'northward_wind_at_100_metres',
       'northward_wind_at_10_metres',
       'precipitation_amount_1hour_Accumulation', 'snow_density',
       'surface_air_pressure']].describe()

# +
#User Inputs
precision=0.5   #Any data between lat/lon +/- precision/2 would be kept in dataframe
lat = 38.0
lon = -119.0
variable = 'snow_density'

#New dataframe for User Inputs
lat_range = [lat+precision/2, lat-precision/2]
lon_range = [lon+precision/2, lon-precision/2]
lat_lon_df = methane_weather_df[(methane_weather_df.lat < lat_range[0]) & 
                                (methane_weather_df.lat > lat_range[1]) & 
                                (methane_weather_df.lon < lon_range[0]) &
                                (methane_weather_df.lon > lon_range[1])]

fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable])
fig1 = plt.title(variable+' over time at lat/lon {}/{}'.format(lat,lon))
fig1=plt.xticks(rotation=45)

# +
#WIND!!
variable1 = 'eastward_wind_at_100_metres'
variable2 = 'eastward_wind_at_10_metres'
variable3 = 'northward_wind_at_100_metres'
variable4 = 'northward_wind_at_10_metres'

#New dataframe for User Inputs
lat_range = [lat+precision/2, lat-precision/2]
lon_range = [lon+precision/2, lon-precision/2]
lat_lon_df = methane_weather_df[(methane_weather_df.lat < lat_range[0]) & 
                                (methane_weather_df.lat > lat_range[1]) & 
                                (methane_weather_df.lon < lon_range[0]) &
                                (methane_weather_df.lon > lon_range[1])]

plt.figure(figsize=(20,10))
fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable1], alpha=0.5)
fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable2], alpha=0.5)
fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable3], alpha=0.5)
fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable4], alpha=0.5)
fig1 = plt.legend([variable1,variable2,variable3,variable4])
fig1 = plt.title('Wind over time at lat/lon {}/{}'.format(lat,lon))
fig1=plt.xticks(rotation=45)

# +
#WIND!!
variable1 = 'eastward_wind_at_100_metres'
variable2 = 'eastward_wind_at_10_metres'
variable3 = 'northward_wind_at_100_metres'
variable4 = 'northward_wind_at_10_metres'

#New dataframe for User Inputs
lat_range = [lat+precision/2, lat-precision/2]
lon_range = [lon+precision/2, lon-precision/2]
lat_lon_df = methane_weather_df[(methane_weather_df.lat < lat_range[0]) & 
                                (methane_weather_df.lat > lat_range[1]) & 
                                (methane_weather_df.lon < lon_range[0]) &
                                (methane_weather_df.lon > lon_range[1])]

plt.figure(figsize=(20,10))
fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable1], alpha=0.5)
fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable2], alpha=0.5)
# fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable3], alpha=0.5)
# fig1 = plt.plot(lat_lon_df.time_utc, lat_lon_df[variable4], alpha=0.5)
fig1 = plt.legend([variable1,variable2,variable3,variable4])
fig1 = plt.title('Wind over time at lat/lon {}/{}'.format(lat,lon))
fig1=plt.xticks(rotation=45)
# -


