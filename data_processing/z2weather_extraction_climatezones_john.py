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

# # ERA5 DATA EXTRACTION & CA CLIMATE ZONE REGIONS
#
# ERA5
#
# Step 1: Extract all unique latitude/longitude combinations from Methane Dataframe. (Used to extract ERA5 data).  
# Step 2: Inspect AWS ERA5 directory and store relevant weather variable names.  
# Step 3: Download raw ERA5 data for weather variables and write to local directory.  
# Step 4: Extract only weather from lat/lon combinations from Methane Dataframe and write to local directory.
# Step 5: 
#
# CA Climate Zones 
#
# Step 1: Create lists of climate zone ID's and polygons.  
# Step 2: Create lists of all lat/lon combinations from the methane dataframe. Convert to a "Point".  
# Step 3: For each "Point", search if point is in a climate zone, if so, save that climate zone and add to methane dataframe.  
# Step 4: Save to S3.

# +
# # !pip install netcdf4
# # !pip install h5netcdf
# # REMEMBER TO RESTART KERNEL

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
    return input_files1


# +
#Pipeline

delete_local_files = False  #set to false if you want to keep .NC raw data in local directory

#For each date in date range we are looking at
for date in date_batches:

    #####################################################################################
    #IMPORT METHANE DATA AND ONLY KEEP THE LAT/LON COMBINATIONS TO EXTRACT FROM ERA5 DATA
    #####################################################################################

    # Read in Methane Data
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder
    file_name=f'{date}_meth.parquet.gzip'
    data_location = 's3://{}/{}'.format(s3_path, file_name)
    methane_df = pd.read_parquet(data_location)
    methane_df['time_utc'] = pd.to_datetime(methane_df['time_utc']).dt.tz_localize(None)
    methane_df['time_utc_hour'] = methane_df['time_utc'].dt.round('h')

    #Create list of lat/lon combinations
    lat_lon = methane_df.groupby(['rn_lat_2', 'rn_lon_2']).count().reset_index()
    locs = []
    for lat,lon in zip(list(lat_lon['rn_lat_2']), list(lat_lon['rn_lon_2'])):
        combined = {}
        combined['name'] = str(lat)+'::'+str(lon)
        combined['lon'] = lon
        combined['lat'] = lat
        locs.append(combined)            #Dictionary of dictionaries of name, longitude, and lat

    #need to convert longitude in [-180,180] (typical range of longitude degrees) into [0,360] (format of ERA5 data)
    for l in locs:
        if l['lon'] < 0:
            l['lon'] = 360 + l['lon']
    
    # create these date variables for later use
    year = date[:4]
    month = date[5:7]
    date_str = date                                  #date string for naming purposes
    import datetime
    date = datetime.date(int(year), int(month), 1)   #datetime to extract weather data
    
    #####################################################################################
    #CHECK EXISTING ERA5 DATA AVAILABILITY
    #####################################################################################
   
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

    #We will skip these variables from our final data
    var_name_list.remove('main')   #just meta data
    var_name_list.remove('sea_surface_wave_from_direction')   #lots of Nan's when joined (since we're focused on california)
    var_name_list.remove('sea_surface_wave_mean_period')     #lots of Nan's when joined (since we're focused on california)
    var_name_list.remove('sea_surface_temperature')     #lots of Nan's when joined (since we're focused on california)
    var_name_list.remove('significant_height_of_wind_and_swell_waves')  #lots of Nan's when joined (since we're focused on california)
    
    #this try is for data before 2019. The format includes a _meta file for EACH variable which is not needed, therefore, remove from our search
    try: 
        meta_names = ['air_pressure_at_mean_sea_level_meta',
                     'air_temperature_at_2_metres_1hour_Maximum_meta', 'air_temperature_at_2_metres_1hour_Minimum_meta',
                     'air_temperature_at_2_metres_meta',
                     'dew_point_temperature_at_2_metres_meta',
                     'eastward_wind_at_100_metres_meta', 'eastward_wind_at_10_metres_meta',
                     'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_meta',
                     'lwe_thickness_of_surface_snow_amount_meta',
                     'northward_wind_at_100_metres_meta', 'northward_wind_at_10_metres_meta',
                     'precipitation_amount_1hour_Accumulation_meta',
                     'sea_surface_temperature_meta',
                     'significant_height_of_wind_and_swell_waves_meta',
                     'snow_density_meta', 'surface_air_pressure_meta',]
        for meta in meta_names:
            var_name_list.remove(meta)
    
    except:
        print("file names do not have an extra word 'meta' in it, we are good to go!")   
    
    
    print("List of variables to download:", var_name_list)  #updated variable list
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    #####################################################################################
    #DOWNLOAD RAW ERA5 DATA AND 
    #WRITE DATAFRAME FOR EACH WEATHER VARIABLE AT SPECIFIED LAT/LON COMBOS FOR ONE DATE
    #####################################################################################    

    start=time.time()

    #For each weather variable
    for var in var_name_list:

        start2=time.time()
        var=str(var)

        #####################################################################################
        #DOWNLOAD DATA FROM AWS BUCKET
        #####################################################################################
        
        # file path patterns for remote S3 objects and corresponding local file
        s3_data_ptrn = '{year}/{month}/data/{var}.nc'
        data_file_ptrn = 'weather_data/{year}{month}_{var}.nc'

        year = date.strftime('%Y')
        month = date.strftime('%m')
        s3_data_key = s3_data_ptrn.format(year=year, month=month, var=var)
        data_file = data_file_ptrn.format(year=year, month=month, var=var)

        if not os.path.isfile(data_file): # check if file already exists, if not, download
            print("Downloading %s from S3..." % s3_data_key)
            client.download_file(era5_bucket, s3_data_key, data_file)
            print("time to download file (seconds): ", time.time()-start2)

        ds = xr.open_dataset(data_file)
        #ds.info

        #####################################################################################
        # TAKE ERA5 DATA AND EXTRACT DATA FROM LAT/LON COMBO
        #####################################################################################

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

        #####################################################################################
        # CONVERT TO DATAFRAME AND WRITE TO PARQUET
        # Format would be a column for dates
        #####################################################################################

        df=ds_locs.to_dataframe()
        df.to_parquet('weather_data/{}_{}.parquet.gzip'.format(date.strftime('%Y_%m'),var), compression='gzip')
        print("time to extract data (seconds): ", time.time()-start2)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("time to download all variables (secs): ", time.time()-start)
    
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

    #Define function for splitting column name (formated as latlong_variablename)
    def split_lat(x):
        return x.split('::')[0]
    def split_lon(x):
        return x.split('::')[1].split("_")[0]

    #Local data directory
    local_path = '/root/methane/data_processing/weather_data/'
    
#     year=str(date)[:4]
#     month=str(date)[5:7]

    #For each data variable, merge to methane data
    for variable in variable_names:

        print("merging variable: ", variable)

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


        
        
#CLEAN UP LOCAL DIRECTORY        
input_files = getInputFiles(local_path)
if delete_local_files:
    for f in input_files:
        os.remove(f)
    print("deleted weather files after all weather merge completed")
    

# +
# # READ FILE TO TEST THAT IT WORKED
# var='2020_12_air_temperature_at_2_metres'
# file_location = 'weather_data/'
# file_name= var + '.parquet.gzip'
# test_df = pd.read_parquet(file_location+file_name)
# test_df
# -



# # CA Climate Zones Addition
# Understand Climate Zones
# * https://cecgis-caenergy.opendata.arcgis.com/datasets/CAEnergy::california-building-climate-zones/explore?location=37.062390%2C-120.193659%2C5.99
# * https://www.pge.com/includes/docs/pdfs/about/edusafety/training/pec/toolbox/arch/climate/california_climate_zones_01-16.pdf

# ### Create column with climate zone regions for each row of the methane dataframe
#
# Step 1: Create lists of climate zone ID's and polygons.  
# Step 2: Create lists of all lat/lon combinations from the methane dataframe. Convert to a "Point".  
# Step 3: For each "Point", search if point is in a climate zone, if so, save that climate zone and add to methane dataframe.  
# Step 4: Save to S3.

# +
#Pipeline

#Read CA climate zone data
file_name = '/root/methane/data_processing/resources/ca_building_climate_zones.geojson'
cl_gdf = gpd.read_file(file_name)

#For each date in date range we are looking at
for date in date_batches:

    start=time.time()
    print("Begin adding regions to date: {}".format(date))
    
    # Read in Methane Weather Data
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder
    file_name=f'{date}_meth_weather.parquet.gzip'
    data_location = 's3://{}/{}'.format(s3_path, file_name)
    df = pd.read_parquet(data_location)
    df['time_utc'] = pd.to_datetime(df['time_utc']).dt.tz_localize(None)
    df['time_utc_hour'] = df['time_utc'].dt.round('h')
    print("Shape:", df.shape)
#     df.head(2)
    
    #Step 1
    #Create mapping, list of all facilities polygons
    zone_id_list = cl_gdf['BZone'].tolist()             #list of climate zones from the geopandas file
    region_poly_list = cl_gdf['geometry'].tolist()      #list of polygons from the geopandas file
#     print("step 1 complete")
    
    #Step 2
    #Create lists of lat/lon combinations and convert to a "Point"
    lats = df['lat'].tolist()   #list of lats from methane df
    lons = df['lon'].tolist()   #list of lons from methane df
    def process_points(lon, lat):
        return Point(lon, lat)
    processed_points = [process_points(lons[i], lats[i]) for i in range(len(lats))]  #list of "Points" for each lat/lon in methane df
#     print("step 2 complete")

    #Step 3
    #For each row, look if the row's "Point" is in a climate zone. If so, that climate zone is now added to a list associate for that row
    point_zones = []   #List of climate zones for each row

    for point in processed_points:
        found=False
        for i, poly in enumerate(region_poly_list, 0):
            if poly.contains(point):
                point_zones.append(zone_id_list[i])
                found = True
                #If point has been found, no need to look at other polys
                break
        if not found:   
            point_zones.append(None)
            
    df['BZone'] = point_zones     #Add list of climate zones for each row to a column in the methane df
#     print("step 3 complete")

    #Step 4
    #Save to S3
    file_name=f'{date}_meth_weather_region.parquet.gzip'
    df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')
    print("Completed adding regions to date in {:.2f} seconds".format(time.time()-start))
    
# -







# -

#IMPORT METHANE DATA
bucket = 'methane-capstone'
subfolder = 'data/pipeline-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='2020-12-30_meth_weather.parquet.gzip'.format(year,month)
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
test_df = pd.read_parquet(data_location)
test_df.head(2)

test_df


