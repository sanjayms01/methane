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

# # MERGE ERA5 WEATHER DATA WITH METHANE DATAFRAME
#
# Step 1:  Import Methane data  
# Step 2:  Create list of all weather variables.  
# Step 3a: For each weather variable, combine all monthly data into one file. Write to local directory.  
# Step 3b: Re-format each weather dataframe and join to methane dataframe. Write methane+weather data to S3 bucket.  

# +
# # !pip install xarray
# # !pip install netcdf4
# # !pip install h5netcdf
# #REMEMBER TO RESTART KERNEL
# -

# Initialize notebook environment.
# %matplotlib inline
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path
import xarray as xr
import time

start=time.time()

# +
#IMPORT METHANE DATA

# Read in Methane Dataframe
bucket = 'methane-capstone'
subfolder = 'data/combined-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='combined-raw.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
methane_df = pd.read_parquet(data_location)
methane_df['time_utc_hour'] = methane_df['time_utc'].dt.round('h')

methane_df.head(2)
# -

# # Append Weather Data
# 1 section per variable.  
#
# For each section:
# 1) First append each month's worth of data into one parquet file  
# 2) Then merge this data into methane dataframe

#Get list of Variable Names
total_files_list = os.listdir('data')
total_files_list.sort()
variable_names = []
for file in total_files_list[1:16]:
    variable_names.append(file[8:])
variable_names

# +
# Set up some global variables
available_data = ['2021.09', '2021.08',  '2021.07',  '2021.06', '2021.05', '2021.04', '2021.03',  '2021.02',  '2021.01',
'2020.12',  '2020.11',  '2020.10',  '2020.09',  '2020.08',  '2020.07', '2020.06',  '2020.05',  '2020.04',  '2020.03',  '2020.02',  '2020.01',
'2019.12',  '2019.11',  '2019.10',  '2019.09',  '2019.08',  '2019.07', '2019.06',  '2019.05',  '2019.04',  '2019.03',  '2019.02',  '2019.01',
'2018.12',  '2018.11']

year_month = '2020.07' ##### UPDATE THIS!!!
file_location = 'data/'

#Define function for splitting data
def split_lat(x):
    return x.split("-")[0]
def split_lon(x):
    y = x.split("-")[1].split("_")[0]
    return "-"+y


# +
# SUBSET METHANE DATAFRAME TO THE MONTH BEING MERGED
year=year_month.split(".")[0]
month=year_month.split(".")[1]
month_plus1={'01':'02', '02':'03', '03':'04', '04':'05','05':'06','06':'07',
            '07':'08', '08':'09', '09':'10', '10':'11','11':'12','12':'01'}
year_plus1={'2018':'2019', '2019':'2020', '2020':'2021', '2021':'2022'}
month2=month_plus1[month]
if month=='12':
    year2=year_plus1[year]
else:
    year2=year
    
filter_date_begin = '{}-{}-01'.format(year,month)
filter_date_end = '{}-{}-01'.format(year2,month2)

methane_df=methane_df[(methane_df['time_utc'] >= filter_date_begin) & (methane_df['time_utc'] < filter_date_end)]
methane_df.head(2)
# -

# ###  Section 1: air_pressure_at_mean_sea_level

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE
variable='air_pressure_at_mean_sea_level'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
  
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head()
# -

# ###  Section 2: air_temperature_at_2_metres

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE
variable='air_temperature_at_2_metres'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour', 'variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 3: air_temperature_at_2_metres_1hour_Maximum

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE
variable='air_temperature_at_2_metres_1hour_Maximum'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()
weather_df=weather_df[weather_df.nv==0]
weather_df.drop(weather_df.columns[1:3],axis=1,inplace=True)  #drop nv and timebound columns 

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time1'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time1'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time1','lat_w','lon_w', 'time0_hour', 'variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 4: air_temperature_at_2_metres_1hour_Minimum

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE
variable='air_temperature_at_2_metres_1hour_Minimum'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()
weather_df=weather_df[weather_df.nv==0]
weather_df.drop(weather_df.columns[1:3],axis=1,inplace=True)  #drop nv and timebound columns 

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time1'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time1'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time1','lat_w','lon_w', 'time0_hour', 'variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 5: dew_point_temperature_at_2_metres

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='dew_point_temperature_at_2_metres'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 6: eastward_wind_at_100_metres

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='eastward_wind_at_100_metres'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour', 'variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 7: eastward_wind_at_10_metres

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='eastward_wind_at_10_metres'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 8: integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation'
weather_df_melt = pd.DataFrame()
variable='integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()
weather_df=weather_df[weather_df.nv==0]
weather_df.drop(weather_df.columns[1:3],axis=1,inplace=True)  #drop nv and timebound columns 

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time1'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time1'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time1','lat_w','lon_w', 'time0_hour', 'variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 9: lwe_thickness_of_surface_snow_amount

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='lwe_thickness_of_surface_snow_amount'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 10: northward_wind_at_100_metres

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='northward_wind_at_100_metres'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 11: northward_wind_at_10_metres

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='northward_wind_at_10_metres'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])

        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 12: precipitation_amount_1hour_Accumulation

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='precipitation_amount_1hour_Accumulation'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()
weather_df=weather_df[weather_df.nv==0]
weather_df.drop(weather_df.columns[1:3],axis=1,inplace=True)  #drop nv and timebound columns 

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time1'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time1'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time1','lat_w','lon_w', 'time0_hour', 'variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 13: snow_density

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='snow_density'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# ###  Section 14: surface_air_pressure

# +
#READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE

variable='surface_air_pressure'
year=year_month.split('.')[0]
month=year_month.split('.')[1]
file_name='{}_{}_'.format(year,month)+variable+'.parquet'
weather_df = pd.read_parquet(file_location+file_name).reset_index()

# Melt columns from different latitudes and longitudes to only 1 column
weather_df_melt = pd.melt(weather_df, id_vars=['time0'], value_vars= weather_df.columns[1:],
                         value_name=file_name.split('.')[0][8:])
        
#Create lat and lon columns for joining to methane data
weather_df_melt['lat_w'] = weather_df_melt['variable'].apply(split_lat)
weather_df_melt['lon_w'] = weather_df_melt['variable'].apply(split_lon)
weather_df_melt[['lat_w','lon_w']] = weather_df_melt[['lat_w','lon_w']].apply(pd.to_numeric)  #need these to be floats

#Truncate datetime for methane dataframe and weather dataframe to 1 hour for merging
weather_df_melt['time0_hour'] = weather_df_melt['time0'].dt.round('h')

#Merge with Methane Data
methane_df = pd.merge(methane_df, weather_df_melt, how='left', left_on = ['time_utc_hour', 'rn_lat', 'rn_lon'], right_on = ['time0_hour','lat_w','lon_w'])
methane_df.drop(['time0','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

print(weather_df_melt.shape)
weather_df_melt.head(2)
# -

# # CHECK MERGED DATA

# +
# methane_df[(methane_df['time_utc'] > '2020-12-01') & (methane_df['time_utc'] < '2021-1-1')]
# -

#write methane data
bucket = 'methane-capstone'
subfolder = 'data/weather/era5'
s3_path_month = bucket+'/'+subfolder
file_name='{}-{}_methane_and_weather.parquet'.format(year,month)
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
methane_df.to_parquet(data_location)
methane_df.head(2)

#IMPORT METHANE DATA
bucket = 'methane-capstone'
subfolder = 'data/weather/era5'
s3_path_month = bucket+'/'+subfolder
file_name='{}-{}_methane_and_weather.parquet'.format(year,month)
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
test_df = pd.read_parquet(data_location)
test_df.head(2)

test_df



print(time.time()-start)