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

# # ERA5 DATA EXTRACTION
#
# Step 1: Extract all unique latitude/longitude combinations from Methane Dataframe. (Used to extract ERA5 data).  
# Step 2: Inspect AWS ERA5 directory and store relevant weather variable names.  
# Step 3a: Download raw ERA5 data and write to local directory.  
# Step 3b: Extract only weather from California and write to local directory.

# +
# # !pip install netcdf4
# # !pip install h5netcdf
# # REMEMBER TO RESTART KERNEL
<<<<<<< HEAD

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

=======
# -

# Initialize notebook environment.
# %matplotlib inline
import boto3
import botocore
import datetime
import matplotlib.pyplot as plt
import os.path
import xarray as xr
import pandas as pd
import numpy as np
import time

# +
#IMPORT METHANE DATA AND ONLY KEEP THE LAT/LON COMBINATIONS TO EXTRACT FROM ERA5 DATA

# Read in Methane Data
bucket = 'methane-capstone'
subfolder = 'data/pipeline-raw-data'
s3_path = bucket+'/'+subfolder
file_name='2020-12-30_meth.parquet.gzip'
data_location = 's3://{}/{}'.format(s3_path, file_name)
methane_df = pd.read_parquet(data_location)
methane_df['time_utc'] = pd.to_datetime(methane_df['time_utc']).dt.tz_localize(None)
methane_df['time_utc_hour'] = methane_df['time_utc'].dt.round('h')
print(methane_df.shape)
methane_df.head()

# +
#Create list of lat/lon combinations
lat_lon = methane_df.groupby(['rn_lat_2', 'rn_lon_2']).count().reset_index()
locs = []
for lat,lon in zip(list(lat_lon['rn_lat_2']), list(lat_lon['rn_lon_2'])):
    combined = {}
    combined['name'] = str(lat)+str(lon)
    combined['lon'] = lon
    combined['lat'] = lat
    locs.append(combined)

for l in locs:
    if l['lon'] < 0:
        l['lon'] = 360 + l['lon']
print(len(locs))
locs[:5]
>>>>>>> 259fa48... added first part of pipeline for methane and weather extraction

# +
from datetime import timedelta, date

start_dt = date(2020, 12, 30 )
end_dt = date(2020, 12, 31)
date_batches = []

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1+timedelta(n)

for dt in daterange(start_dt, end_dt):
    date_batches.append(dt.strftime("%Y-%m-%d"))

# print(date_batches)

<<<<<<< HEAD
# date_list=[]
# year_month_set=set()

# for date in date_batches:
#     year = date[:4]
#     month = date[5:7]
#     year_month = year+month
#     if year_month not in year_month_set:
#         date_list.append(datetime.date(int(year),int(month),1))
#         year_month_set.add(year_month)
    
# # print(date_list)
# print(year_month_set)

# +
# #IMPORT METHANE DATA AND ONLY KEEP THE LAT/LON COMBINATIONS TO EXTRACT FROM ERA5 DATA

# # Read in Methane Data
# bucket = 'methane-capstone'
# subfolder = 'data/pipeline-raw-data'
# s3_path = bucket+'/'+subfolder
# file_name=f'{date_batches[0]}_meth.parquet.gzip'
# data_location = 's3://{}/{}'.format(s3_path, file_name)
# methane_df = pd.read_parquet(data_location)
# methane_df['time_utc'] = pd.to_datetime(methane_df['time_utc']).dt.tz_localize(None)
# methane_df['time_utc_hour'] = methane_df['time_utc'].dt.round('h')
# print(methane_df.shape)
# methane_df.head()

# +
# #Create list of lat/lon combinations
# lat_lon = methane_df.groupby(['rn_lat_2', 'rn_lon_2']).count().reset_index()
# locs = []
# for lat,lon in zip(list(lat_lon['rn_lat_2']), list(lat_lon['rn_lon_2'])):
#     combined = {}
#     combined['name'] = str(lat)+str(lon)
#     combined['lon'] = lon
#     combined['lat'] = lat
#     locs.append(combined)

# for l in locs:
#     if l['lon'] < 0:
#         l['lon'] = 360 + l['lon']
# print(len(locs))
# locs[:5]

# +
#Helper Function

def getInputFiles(local_path):
    '''
    Get list of input files stored on Sagemaker directory 
    (run after getNCFile helper function)
    '''
    input_files1 = sorted(list(iglob(join(local_path, '**', '**.nc' ), recursive=True)), reverse=True)
    input_files2 = sorted(list(iglob(join(local_path, '**', '**.gzip' ), recursive=True)), reverse=True)
    input_files = input_files1+input_files2
    return input_files


# +
delete_local_files = True

for date in date_batches:

    #IMPORT METHANE DATA AND ONLY KEEP THE LAT/LON COMBINATIONS TO EXTRACT FROM ERA5 DATA

    # Read in Methane Data
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder
    file_name=f'{date_batches[0]}_meth.parquet.gzip'
    data_location = 's3://{}/{}'.format(s3_path, file_name)
    methane_df = pd.read_parquet(data_location)
    methane_df['time_utc'] = pd.to_datetime(methane_df['time_utc']).dt.tz_localize(None)
    methane_df['time_utc_hour'] = methane_df['time_utc'].dt.round('h')
#     print(methane_df.shape)
#     methane_df.head()
    
    #Create list of lat/lon combinations
    lat_lon = methane_df.groupby(['rn_lat_2', 'rn_lon_2']).count().reset_index()
    locs = []
    for lat,lon in zip(list(lat_lon['rn_lat_2']), list(lat_lon['rn_lon_2'])):
        combined = {}
        combined['name'] = str(lat)+str(lon)
        combined['lon'] = lon
        combined['lat'] = lat
        locs.append(combined)

    for l in locs:
        if l['lon'] < 0:
            l['lon'] = 360 + l['lon']
#     print(len(locs))
#     locs[:5]
    
    # update string date to datetime date
    year = date[:4]
    month = date[5:7]
    date_str = date
    import datetime
    date = datetime.date(int(year), int(month), 1)
    
=======
date_list=[]
year_month_set=set()

for date in date_batches:
    year = date[:4]
    month = date[5:7]
    year_month = year+month
    if year_month not in year_month_set:
        date_list.append(datetime.date(int(year),int(month),1))
        year_month_set.add(year_month)
    
# print(date_list)
print(year_month_set)
# -

for date in date_list:
>>>>>>> 259fa48... added first part of pipeline for methane and weather extraction
    #CHECK ERA5 DATA

    #bucket
    era5_bucket = 'era5-pds'

    # No AWS keys required
    client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    #See what data is available for date
    keys = []
    var_name_list = []              #STORE ALL THE WEATHER VARIABLES IN THIS LIST
    prefix = date.strftime('%Y/%m/')

    response = client.list_objects_v2(Bucket=era5_bucket, Prefix=prefix)
    response_meta = response.get('ResponseMetadata')

    if response_meta.get('HTTPStatusCode') == 200:
        contents = response.get('Contents')
        if contents == None:
            print("No objects are available for %s" % date.strftime('%B, %Y'))
        else:
            for obj in contents:
                keys.append(obj.get('Key'))
            print("There are %s objects available for %s\n--" % (len(keys), date.strftime('%B, %Y')))
            for k in keys:
                print(k)
                var_name_list.append(k.split("/")[-1].split('.')[0])
    else:
        print("There was an error with your request.")

    #These variables are not relevant to our data
    var_name_list.remove('main')   #just meta data
    var_name_list.remove('sea_surface_wave_from_direction')   #lots of Nan's when joined (since we're focused on california)
    var_name_list.remove('sea_surface_wave_mean_period')     #lots of Nan's when joined (since we're focused on california)
    var_name_list.remove('sea_surface_temperature')     #lots of Nan's when joined (since we're focused on california)
    var_name_list.remove('significant_height_of_wind_and_swell_waves')  #lots of Nan's when joined (since we're focused on california)
    
    
    try: 
        meta_names = ['air_pressure_at_mean_sea_level_meta',
                     'air_temperature_at_2_metres_1hour_Maximum_meta',
                     'air_temperature_at_2_metres_1hour_Minimum_meta',
                     'air_temperature_at_2_metres_meta',
                     'dew_point_temperature_at_2_metres_meta',
                     'eastward_wind_at_100_metres_meta',
                     'eastward_wind_at_10_metres_meta',
                     'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_meta',
                     'lwe_thickness_of_surface_snow_amount_meta',
                     'northward_wind_at_100_metres_meta',
                     'northward_wind_at_10_metres_meta',
                     'precipitation_amount_1hour_Accumulation_meta',
                     'sea_surface_temperature_meta',
                     'significant_height_of_wind_and_swell_waves_meta',
                     'snow_density_meta',
                     'surface_air_pressure_meta',
                     'test']
        for meta in meta_names:
            var_name_list.remove(meta)
    
    except:
        print("file names do not have an extra word 'meta' in it, we are good to go!")   
    
    
    print(var_name_list)  #updated variable list
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    #WRITE DATAFRAME FOR EACH WEATHER VARIABLE AT SPECIFIED LAT/LON COMBOS FOR ONE MONTH

    start=time.time()
    # select date and variable of interest
    # date = datetime.date(2021,9,1)  #ALREADY SPECIFIED ABOVE

    for var in var_name_list:

        start2=time.time()
        var=str(var)

        #DOWNLOAD DATA FROM AWS BUCKET

        # file path patterns for remote S3 objects and corresponding local file
        s3_data_ptrn = '{year}/{month}/data/{var}.nc'
        data_file_ptrn = 'weather_data/{year}{month}_{var}.nc'

        year = date.strftime('%Y')
        month = date.strftime('%m')
        s3_data_key = s3_data_ptrn.format(year=year, month=month, var=var)
        data_file = data_file_ptrn.format(year=year, month=month, var=var)

        if not os.path.isfile(data_file): # check if file already exists
            print("Downloading %s from S3..." % s3_data_key)
            client.download_file(era5_bucket, s3_data_key, data_file)
            print("time to download file (seconds): ", time.time()-start2)

        ds = xr.open_dataset(data_file)
        #ds.info

        # TAKE ERA5 DATA AND EXTRACT DATA FROM LAT/LON COMBO

        ds_locs = xr.Dataset()
        # interate through the locations and create a dataset
        # containing the weather values for each location
        print("Start Extraction: ", var)
        start3=time.time()
        for l in locs:
            name = l['name']+'_'+var
            lon = l['lon']
            lat = l['lat']
            var_name = name

            ds2 = ds.sel(lon=lon, lat=lat, method='nearest')

            lon_attr = '%s_lon' % name
            lat_attr = '%s_lat' % name

            ds2.attrs[lon_attr] = ds2.lon.values.tolist()
            ds2.attrs[lat_attr] = ds2.lat.values.tolist()
            ds2 = ds2.rename({var : var_name}).drop(('lat', 'lon'))

            ds_locs = xr.merge([ds_locs, ds2])

        # CONVERT TO DATAFRAME AND WRITE TO PARQUET
        df=ds_locs.to_dataframe()
        df.to_parquet('weather_data/{}_{}.parquet.gzip'.format(date.strftime('%Y_%m'),var), compression='gzip')
        print("time to extract data (seconds): ", time.time()-start2)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("time to download all variables (secs): ", time.time()-start)
<<<<<<< HEAD
    
    ######################################
    # MERGE WEATHER DATA WITH METHANE DATA
    ######################################

    print("Start merging weather data to methane data, for each date")
    merge_start = time.time()
    
    #List of Variable Names:
    variable_names = ['air_pressure_at_mean_sea_level',
    'air_temperature_at_2_metres',
    'air_temperature_at_2_metres_1hour_Maximum',
    'air_temperature_at_2_metres_1hour_Minimum',
    'dew_point_temperature_at_2_metres',
    'eastward_wind_at_100_metres',
    'eastward_wind_at_10_metres',
    'lwe_thickness_of_surface_snow_amount',
    'northward_wind_at_100_metres',
    'northward_wind_at_10_metres',
    'snow_density',
    'surface_air_pressure',             
    'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
    'precipitation_amount_1hour_Accumulation']

    #These variables have a different dataframe format, need additional filtering
    variable_names_additional_filter=['integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
    'precipitation_amount_1hour_Accumulation', 'air_temperature_at_2_metres_1hour_Maximum','air_temperature_at_2_metres_1hour_Minimum' ]

    #Define function for splitting data
    def split_lat(x):
        return x.split("-")[0]
    def split_lon(x):
        y = x.split("-")[1].split("_")[0]
        return "-"+y

    #Local data
    local_path = '/root/methane/data_processing/weather_data/'
    
#     year=str(date)[:4]
#     month=str(date)[5:7]

    for variable in variable_names:

        print("working on variable: ", variable)

        #READ WEATHER DF STORED LOCALLY, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO S3 DRIVE
        file_name='{}_{}_'.format(year,month)+variable+'.parquet.gzip'
        weather_df = pd.read_parquet(local_path+file_name).reset_index()

        if variable in variable_names_additional_filter:
            print('feature w/ additional filter')
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
            methane_df.drop(['time1','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns

        else: 
            print('feature w/o additional filter')
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
        
        #write methane_weather data
#         bucket = 'methane-capstone'
#         subfolder = 'data/pipeline-raw-data'
#         s3_path = bucket+'/'+subfolder
        file_name=f'{date_str}_meth_weather.parquet.gzip'
        methane_df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')
        
        print("date: {}, methane and weather data merged successfully in {} seconds".format(date, (time.time()- merge_start)))


    input_files = getInputFiles(local_path)
    if delete_local_files:
        for f in input_files:
            os.remove(f)
        print("deleted weather files after all weather merge completed")
    
=======
>>>>>>> 259fa48... added first part of pipeline for methane and weather extraction

# +
# # READ FILE TO TEST THAT IT WORKED
# var='2020_12_air_temperature_at_2_metres'
# file_location = 'weather_data/'
# file_name= var + '.parquet.gzip'
# test_df = pd.read_parquet(file_location+file_name)
# test_df
# -



# # MERGE ERA5 WEATHER DATA WITH METHANE DATAFRAME

<<<<<<< HEAD
# +
# #Create combined dataframe
# combined_df = pd.DataFrame()
# file_name=f'{str(start_dt)}_{str(end_dt)}_meth.parquet.gzip'
# combined_df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip') 


# #combine each batch df into a single df and write to S3
# if combined_df.shape[0] == 0:
#     combined_df = cur_batch_df
# else:
#     combined_df = combined_df.append(cur_batch_df)
#     combined_df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip') 

# -

=======
>>>>>>> 259fa48... added first part of pipeline for methane and weather extraction
methane_df.head()

# ### Append Weather Data
#
# Steps:
# 1) Format extracted weather data into a format to merge to methane dataframe  
# 2) Merge formatted data into methane dataframe

<<<<<<< HEAD
# ###  Weather Variables

# +
#write methane data

methane_df.to_parquet(data_location)
methane_df.head(2)
# -

#IMPORT METHANE DATA
bucket = 'methane-capstone'
subfolder = 'data/pipeline-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='2020-12-31_meth_weather.parquet.gzip'.format(year,month)
=======
# +
#List of Variable Names:
variable_names = ['air_pressure_at_mean_sea_level',
'air_temperature_at_2_metres',
'air_temperature_at_2_metres_1hour_Maximum',
'air_temperature_at_2_metres_1hour_Minimum',
'dew_point_temperature_at_2_metres',
'eastward_wind_at_100_metres',
'eastward_wind_at_10_metres',
'lwe_thickness_of_surface_snow_amount',
'northward_wind_at_100_metres',
'northward_wind_at_10_metres',
'snow_density',
'surface_air_pressure',             
'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
'precipitation_amount_1hour_Accumulation']

#These variables have a different dataframe format, need additional filtering
variable_names_additional_filter=['integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
'precipitation_amount_1hour_Accumulation', 'air_temperature_at_2_metres_1hour_Maximum','air_temperature_at_2_metres_1hour_Minimum' ]

#Define function for splitting data
def split_lat(x):
    return x.split("-")[0]
def split_lon(x):
    y = x.split("-")[1].split("_")[0]
    return "-"+y

#Local data
file_location = 'weather_data/'

#Specify time
date_list
date = str(date_list[0])
date
# -

# ###  Weather Variables

# +
year=date[:4]
month=date[5:7]

for variable in variable_names:
    
    print("working on variable: ", variable)
    
    #READ WEATHER DF, REFORMAT, MERGE WITH METHANE DATAFRAME, WRITE TO DRIVE
    file_name='{}_{}_'.format(year,month)+variable+'.parquet.gzip'
    weather_df = pd.read_parquet(file_location+file_name).reset_index()

    if variable in variable_names_additional_filter:
        print('additional filter')
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
        methane_df.drop(['time1','lat_w','lon_w', 'time0_hour','variable'], axis=1,inplace=True) #drop irrelvant columns
    
    else: 
        print('no additional filter')
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



    
# -

methane_df

# # CHECK MERGED DATA

methane_df['time_utc'] =methane_df['time_utc'].astype('datetime64[ms]')  #convert to miliseconds b/c ns doesnt work for writing for some reason...
methane_df

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
>>>>>>> 259fa48... added first part of pipeline for methane and weather extraction
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
test_df = pd.read_parquet(data_location)
test_df.head(2)

<<<<<<< HEAD
test_df
=======

>>>>>>> 259fa48... added first part of pipeline for methane and weather extraction


