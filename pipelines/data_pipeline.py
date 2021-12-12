# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (Data Science)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/datascience-1.0
# ---

# # DATA PIPELINE (EXTRACTING METHANE, WEATHER AND CLIMATE ZONE DATA)

# ## Imports and Installs

# +
# # Run following pip installs and restart notebook
# # !pip install xarray
# # !pip install geopandas
# # !pip install shapely
# # !pip install netCDF4
# # REMEMBER TO RESTART NOTEBOOK
# -

# s5p good until 1 day before  
# era5 good until 18-19 days before...
#
#
#
# nov 1st data available 
# nov 19th.

# +
# Imports
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
#     import plotly.express as px
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
        
# alt.data_transformers.disable_max_rows()


# -

# ## Define Date Range for Generating Dataset  
#
# Define date range (`start_dt` and `end_dt`).
#

# +
#Step 1: Define date range
from datetime import timedelta, date

start_dt = date(2021, 10, 22 )          #DEFINE START DATE HERE
end_dt = date(2021, 10, 25)              #DEFINE END DATE HERE

localtime = time.localtime(time.time())
# print("Local current time :", localtime)
y = localtime.tm_year
m = localtime.tm_mon
d = localtime.tm_mday
# n = 7                                 # number of days to look back
# end_dt = date(y, m, d)              #If you want current date
# start_dt = end_dt-timedelta(n)

date_batches = []
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1+timedelta(n)
for dt in daterange(start_dt, end_dt):
    date_batches.append(dt.strftime("%Y-%m-%d"))
print(date_batches)                     #CHECK IF THIS LIST INCLUDES ALL DATES IN DEFINED DATERANGE


#Read current dataframe from s3
bucket = 'methane-capstone'
subfolder = 'data/dt=latest'
s3_path = bucket+'/'+subfolder 
file_name='data-zone-combined.parquet.gzip'
data_location = 's3://{}/{}'.format(s3_path, file_name)
current_df = pd.read_parquet(data_location)

print("\n \n Last row of current complete dataframe has date of: ", current_df.time_utc.iloc[-1].strftime("%Y-%m-%d"))
print("start_dt should be: ", (current_df.time_utc.iloc[-1]+ timedelta(days=1)).strftime("%Y-%m-%d"), "\n \n")


if current_df.time_utc.iloc[-1].strftime("%Y-%m-%d") in date_batches:
    print("#######################################")
    print("########## WARNING ####################")
    print("#######################################")
    print("")
    print("Last row of dataframe is within the date range specified above, please correct date range before running this notebook!")
    print("Or else, we could have duplicate data in dataframe!")
    print("")
    print("#######################################")
    print("########## END WARNING ################")
    print("#######################################")

del current_df  #delete variable after task complete

# -

# ## Methane S5P Extraction
# Goal: To extract methane reading data from Sentinel 5P via Amazon AWS public database.  
# Steps:  
# 1. Specify Region of Interest, current region is California  
# 2. Download whole batch (each date worth of .NC files) to local.    
# 3. Parse each file in batch, and build a dataframe for each batch  
# 4. For every batch we will flush the data to a `.parquet.gzip` straight to S3 in respective folder.
#     * Format `{date}_meth.parquet.gzip`
# 6. Delete contents inside Sagemaker batch folder where `.nc` files were downloaded for a respective batch
# 7. Write out a `.txt` file that writes out all the steps that happened in the pipeline  

def start_s5p_extraction(date_batches):  
    #Define Region of Interest, California
    ca_geo_path = "/root/methane/data_processing/resources/california.geojson"
    ca_geo_file = open(ca_geo_path)
    ca_gpd = gpd.read_file(ca_geo_file)
    cali_polygon = ca_gpd['geometry'][0]

    #### HELPER FUNCTIONS ####
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    def getNCFile(date, local_path, outF):
        '''Function to get a S5P file from AWS S3 to local'''
        start = time.time()
        try:
            # Copy all the files from the entire folder for specified date
            year = date[:4]
            month = date[5:7]
            day = date[-2:]
            command = ['aws','s3','cp', 
                       f's3://meeo-s5p/OFFL/L2__CH4___/{year}/{month}/{day}', 
                       local_path, '--recursive']
            subprocess.check_output(command)
        except:
            print_write(f"ISSUE GETTING: {batch_file_name}", outF)
        end = time.time()
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    def getInputFiles(local_path):
        '''
        Get list of input files stored on Sagemaker directory 
        (run after getNCFile helper function)
        '''
        input_files = sorted(list(iglob(join(local_path, '**', '*CH4*.nc' ), recursive=True)), reverse=True)
        return input_files
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    def loadNCFiles(input_files, outF):
        '''Use xarray to load .nc file S5p products'''    
        start = time.time()
        s5p_products = {}
        for file_name in input_files:
            name = file_name.split('/')[-1]
            start_date, end_date = list(filter(lambda x: len(x) == 15, file_name.split("_")))
            key = start_date + '::' + end_date
            try:
                #Open product - PRODUCT
                with xr.load_dataset(file_name, group='PRODUCT') as s5p_prod:
                    s5p_products[key] = s5p_prod
                s5p_prod.close()
            except:
                print_write(f"loadNCFiles - FAILED: {key}", outF)
        end = time.time()
        return s5p_products, getHumanTime(end-start)
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    def filter_in_cali(row):
        '''Filter apply function for CA'''
        point = Point(row['lon'], row['lat'])
        return cali_polygon.contains(point)
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    def processFiles(s5p_products, qa_thresh, outF):
        '''Process all files that are in keys in `s5p_products` '''
        columns = ['time_utc', 
                   'lat', 'lon', 
                   'rn_lat_1','rn_lon_1',
                   'rn_lat_2','rn_lon_2',
                   'rn_lat_5','rn_lon_5',
                   'rn_lat','rn_lon',
                   'qa_val', 
                   'methane_mixing_ratio',
                   'methane_mixing_ratio_precision',
                   'methane_mixing_ratio_bias_corrected']
        df_ca = pd.DataFrame(columns=columns)

        init_start = time.time()

        first_orbit = 0        #keep track of the first orbit with data in California
        last_orbit  = 0        #keep track of the last orbit with data in California, stop extracting other orbits afterwards

        for file_number, product_key in enumerate(s5p_products, 1):

            # If the last orbit to scan California has been searched,
            # stop extract data from rest of the orbit data
            # Note: there are 14 orbits per day
            if last_orbit == 1:
                break

            start_time = time.time()
            s5p_prod = s5p_products[product_key]
            df_cur_scan = pd.DataFrame(columns=columns)
            times_utc = np.array(s5p_prod.time_utc[0, :])

            print_write(f'\tfile_num: {file_number} - {product_key}. start_parse.', outF)

            times_array = []
            for utc, qa in zip(times_utc, s5p_prod.qa_value[0, :, :]):
                times_array.extend([utc] * len(qa))

            lats = np.array(s5p_prod.latitude[0, :, :]).ravel()
            lons = np.array(s5p_prod.longitude[0, :, :]).ravel()
            qa_vals = np.nan_to_num(s5p_prod.qa_value[0, :, :], nan=-9999).ravel()
            methane_mixing_ratio = np.nan_to_num(s5p_prod.methane_mixing_ratio[0, :, :], nan=-9999).ravel()
            methane_mixing_ratio_precision = np.nan_to_num(s5p_prod.methane_mixing_ratio_precision[0, :, :], nan=-9999).ravel()
            methane_mixing_ratio_bias_corrected = np.nan_to_num(s5p_prod.methane_mixing_ratio_bias_corrected[0, :, :], nan=-9999).ravel()

            cur_ts_dict = {
                'time_utc' : times_array,
                'lat' : lats,
                'lon' : lons,
                'rn_lat_1': np.round(lats, 1),
                'rn_lon_1': np.round(lons, 1),
                'rn_lat_2': np.round(lats*5)/5,
                'rn_lon_2': np.round(lons*5)/5,
                'rn_lat_5':  np.round(lats*2)/2,
                'rn_lon_5': np.round(lons*2)/2,
                'rn_lat': np.round(lats, 0),
                'rn_lon': np.round(lons, 0),
                'qa_val' : qa_vals,
                'methane_mixing_ratio' : methane_mixing_ratio,
                'methane_mixing_ratio_precision' : methane_mixing_ratio_precision,
                'methane_mixing_ratio_bias_corrected' : methane_mixing_ratio_bias_corrected,
            }

            df_cur_ts = pd.DataFrame(cur_ts_dict)
            df_cur_scan = pd.concat([df_cur_scan, df_cur_ts], ignore_index=True)   

            #QA Mask
            qa_mask_df = df_cur_scan['qa_val'] >= qa_thresh

            #Methane Mask
            meth_ca_mask_df = (df_cur_scan.methane_mixing_ratio != -9999) & \
                              (df_cur_scan.methane_mixing_ratio_precision != -9999) & \
                              (df_cur_scan.methane_mixing_ratio_bias_corrected != -9999)

            #California Geo Mask
            cali_mask = df_cur_scan.apply(filter_in_cali, axis=1)

            #Join Masks
            mask_join_df = qa_mask_df & cali_mask & meth_ca_mask_df
            df_cur_scan_masked = df_cur_scan[mask_join_df]

            df_ca = pd.concat([df_ca, df_cur_scan_masked], ignore_index=True)
            end_time = time.time()
            print_write(f'\t\t\t\t\tend_parse. shape: {df_cur_scan_masked.shape}, time_taken: {getHumanTime(end_time-start_time)}', outF)

            if df_cur_scan_masked.shape[0] > 0:
                first_orbit = 1

            if first_orbit == 1 and df_cur_scan_masked.shape[0] == 0:
                last_orbit = 1
                print("last orbit over region reached! start extracting next day!")

        return df_ca, getHumanTime(end_time-init_start)

    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 

    #### CONFIG ####
    local_pipe_log_path = '/root/methane/data_processing/pipeline_runs/'
    local_path = '/root/methane/data_processing/nc_data/cur_batch/'
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder

    pipeline_start = True       #do we want to start this cell or not
    delete_local_files = True   #do we want the local files deleted after each batch? or keep everything
    qa_thresh = 0.0             #do we want to keep specific quality ratings of sp5 methane data?

    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
    ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 

    #### S5P Pipeline ####

    if pipeline_start:

        #Give the pipeline log file a name
        pipe_start = str(date_batches[0])   

        try:
            os.makedirs(local_pipe_log_path)
        except:
            print("pipeline logs folder already exist, we're good!")

        outF = open(f"{local_pipe_log_path}pipe_log_{pipe_start}.txt", "w")
        print_write("######### PIPELINE STARTED #########", outF)

        init = time.time()    

        try:
            for batch_num, date in enumerate(date_batches, 1):

                print_write(f"\nbatch_num: {batch_num} - start download", outF)
                batch_dn_start = time.time()

                #Download all 14 .NC files on that one date
                try:
                    os.makedirs(local_path)
                except:
                    print("folder for raw nc files already exist, we're good!")
                getNCFile(date, local_path, outF)

                batch_dn_end = time.time()
                print_write(f"batch_num: {batch_num} - finish download, time_taken: {getHumanTime(batch_dn_end-batch_dn_start)}", outF)

                #Get list of input files
                input_files = getInputFiles(local_path)

                #Load NC Files into dictionary
                print_write(f"batch_num: {batch_num} - start NC files load", outF)
                s5p_products, load_nc_time = loadNCFiles(input_files, outF)
                print_write(f"batch_num: {batch_num} - end NC files load, time_taken: {load_nc_time}", outF)

                #One entire batch is processsed into `cur_df`
                #We name the `cur_df` as `cur_batch_df` for now
                cur_batch_df, process_time = processFiles(s5p_products, qa_thresh, outF)

                ### Write `cur_batch_df` for each batch to S3 ###
                try:
                    file_name=f'{date}_meth.parquet.gzip'
                    cur_batch_df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')

                except:
                    write_loc = 's3://{}/{}'.format(s3_path+f'/{year}', file_name)
                    print_write(f"ERROR S3 WRITE: {write_loc}", outF)

                if delete_local_files:
                    for f in input_files:
                        os.remove(f)
                    print_write("deleted nc files", outF)

                batch_end_time = time.time()

                print_write(f"batch_num: {batch_num} batch_total_time: {getHumanTime(batch_end_time - batch_dn_start)} download_time: {getHumanTime(batch_dn_end-batch_dn_start)}, load_nc_time: {load_nc_time}, process_time: {process_time}, cur_batch_df_shape: {cur_batch_df.shape}", outF)
                print_write("#######################", outF)
            fin = time.time()
            print_write(f"\nTOTAL PIPELINE TIME: {getHumanTime(fin-init)}", outF)
            outF.close()

        except:
            print_write(f"\n !!KABOOM!!", outF)
            outF.close()

    else:
        print("METHANE PIPELINE CLOSED")


# ## Weather ERA5 Extraction  
# Goal: To extract weather data from ERA5 via Amazon AWS public database.  
# Steps:  
# Step 1: Extract all unique latitude/longitude combinations from Methane Dataframe. (Used to extract ERA5 data).  
# Step 2: Inspect AWS ERA5 directory and store relevant weather variable names.  
# Step 3: Download raw ERA5 data for weather variables and write to local directory.  
# Step 4: Extract only weather from lat/lon combinations from Methane Dataframe and write to local directory.
# Step 5: Merge weather data with methane data. 
#

def start_era5_extraction(date_batches):  

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
            print("")
    #         print("file names do not have an extra word 'meta' in it, we are good to go!")   

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

            try:
                os.makedirs('weather_data')
            except:
                print("folder for storing weather data already exist, we're good!")

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

            print("date: {}, methane and weather data merged successfully in {} seconds".format(date_str, (time.time()- merge_start)))

    #CLEAN UP LOCAL DIRECTORY       
    input_files1 = sorted(list(iglob(join(local_path, '**', '**.nc' ), recursive=True)), reverse=True)
    input_files2 = sorted(list(iglob(join(local_path, '**', '**.gzip' ), recursive=True)), reverse=True)
    input_files = input_files1+input_files2
    if delete_local_files:
        for f in input_files:
            os.remove(f)
        print("deleted weather files after all weather merge completed")

    print("WEATHER PIPELINE CLOSED")


# ## CA Climate Zone Addition  
# Goal: Create column with CA climate zone regions for each row of methane dataframe. 
#
# Understand Climate Zones
#
# https://cecgis-caenergy.opendata.arcgis.com/datasets/CAEnergy::california-building-climate-zones/explore?location=37.062390%2C-120.193659%2C5.99
# https://www.pge.com/includes/docs/pdfs/about/edusafety/training/pec/toolbox/arch/climate/california_climate_zones_01-16.pdf  
#
# Steps:  
# Step 1: Create lists of climate zone ID's and polygons.  
# Step 2: Create lists of all lat/lon combinations from the methane dataframe. Convert to a "Point".  
# Step 3: For each "Point", search if point is in a climate zone, if so, save that climate zone and add to methane dataframe.  
# Step 4: Save to S3.
#

def start_CACZ_addition(date_batches):  
    
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

        #Step 1 Create mapping, list of all facilities polygons
        zone_id_list = cl_gdf['BZone'].tolist()             #list of climate zones from the geopandas file
        region_poly_list = cl_gdf['geometry'].tolist()      #list of polygons from the geopandas file

        #Step 2 Create lists of lat/lon combinations and convert to a "Point"
        lats = df['lat'].tolist()   #list of lats from methane df
        lons = df['lon'].tolist()   #list of lons from methane df
        def process_points(lon, lat):
            return Point(lon, lat)
        processed_points = [process_points(lons[i], lats[i]) for i in range(len(lats))]  #list of "Points" for each lat/lon in methane df

        #Step 3 For each row, look if the row's "Point" is in a climate zone. If so, that climate zone is now added to a list associate for that row
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

        #Step 4 Save to S3
        file_name=f'{date}_meth_weather_region.parquet.gzip'
        df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')
        print("Completed adding regions to date in {:.2f} seconds".format(time.time()-start))


# ## Data Pre Processing  
# Goal: Input each date's data and merge into one dataframe. Pre process data by grouping by date per CA Climate Zone.   
#
# Steps:  
# Step 1: Import each date's dataframe and groupby date for EACH zone.    
# Step 2: Merge grouped data for each zone into one dataframe (for each date).   
# Step 3: Merge each date's (non-grouped) data into one dataframe. Merge each date's grouped data into one dataframe.  
# Step 4: Save both dataframes to S3.  
#

def data_pre_process(date_batches):

    #### HELPER FUNCTIONS ####
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
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
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    def get_processed_df(df, zone):

        cur_df = df[df['BZone'] == zone]
        groups, bins = get_time_resolution_groups(cur_df)
        columns = groups[0].columns.tolist()
        df_final = pd.DataFrame(columns=columns)

        for ind, group in enumerate(groups, 1):
            df_final = pd.concat([df_final, group])

        df_final = df_final.sort_values('time_utc')
        df_final['BZone'] = zone
        df_final['BZone'] = df_final['BZone'].astype(int)
        return df_final
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################


    #Pipeline
    bucket = 'methane-capstone'
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder

    grouped_combined = pd.DataFrame()   #for saving grouped by zone data
    address_combined = pd.DataFrame()   #for saving all lat/lon data

    for date in date_batches:

        start=time.time()
        print("Begin grouping data for date: {}".format(date))

        # Read in Methane Data
        file_name=f'{date}_meth_weather_region.parquet.gzip'
        data_location = 's3://{}/{}'.format(s3_path, file_name)
        df = pd.read_parquet(data_location)
        
        #Combine non-grouped data into one dataframe
        df_address = df
        df_address['time_utc'] = pd.to_datetime(df_address['time_utc']).dt.tz_localize(None)   
        if address_combined.shape[0] == 0:
            address_combined = df_address
        else:
            address_combined = address_combined.append(df_address)   
            
        #Group data by zones and save to S3    
        ZONES = ['1','2','3','4','5','6','7','8', 
                 '9','10','11','12','13','14','15','16']

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

        #Combine grouped data into one dataframe
        df_grouped = regions_combined
        df_grouped['time_utc'] = pd.to_datetime(df_grouped['time_utc']).dt.tz_localize(None)

        if grouped_combined.shape[0] == 0:
            grouped_combined = df_grouped
        else:
            grouped_combined = grouped_combined.append(df_grouped)     
            

    grouped_combined = grouped_combined[['time_utc', 'BZone', 'reading_count', 'methane_mixing_ratio_mean',
           'lat_mean', 'methane_mixing_ratio_precision_mean',
           'methane_mixing_ratio_bias_corrected_mean',
           'air_pressure_at_mean_sea_level_mean',
           'air_temperature_at_2_metres_mean',
           'air_temperature_at_2_metres_1hour_Maximum_mean',
           'air_temperature_at_2_metres_1hour_Minimum_mean',
           'dew_point_temperature_at_2_metres_mean',
           'eastward_wind_at_100_metres_mean', 'eastward_wind_at_10_metres_mean',
           'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean',
           'lwe_thickness_of_surface_snow_amount_mean',
           'northward_wind_at_100_metres_mean', 'northward_wind_at_10_metres_mean',
           'precipitation_amount_1hour_Accumulation_mean', 'snow_density_mean',
           'surface_air_pressure_mean', 'qa_val_mode', 'qa_val_mean']]
    
    address_combined = address_combined[['time_utc', 'lat', 'lon', 'rn_lat_1', 'rn_lon_1',
       'rn_lat_2', 'rn_lon_2', 'rn_lat_5', 'rn_lon_5', 'rn_lat', 'rn_lon',
       'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision',
       'methane_mixing_ratio_bias_corrected', 'time_utc_hour',
       'air_pressure_at_mean_sea_level', 'air_temperature_at_2_metres',
       'air_temperature_at_2_metres_1hour_Maximum',
       'air_temperature_at_2_metres_1hour_Minimum',
       'dew_point_temperature_at_2_metres', 'eastward_wind_at_100_metres',
       'eastward_wind_at_10_metres',
       'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
       'lwe_thickness_of_surface_snow_amount', 'northward_wind_at_100_metres',
       'northward_wind_at_10_metres',
       'precipitation_amount_1hour_Accumulation', 'snow_density',
       'surface_air_pressure', 'BZone']]
       
    #save to s3
    start = date_batches[0]
    end = date_batches[-1]
    file_name=f'merged_{start}_{end}_data-zone-combined.parquet.gzip'
    grouped_combined.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')     
    
    file_name=f'merged_{start}_{end}_data-address-combined.parquet.gzip'
    address_combined.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')  


# ## Merge to Final Dataframe
# Goal: Merge new downloaded data to final dataframe. 
#
# Steps:  
# Step 1: Import existing final dataframe.  
# Step 2: Import pre-processed new data created recently from previous steps in this notebook.       
# Step 3: Check if all of the weather data columns are NaN's, if so, drop those rows (b/c ERA5 weather data hasn't been updated yet on Amazon Public Data Registry).    
# Step 4: Merge two dataframes.  
# Step 5: Save to S3 (replace old final dataframe and save to archive folder for backup).      
#

# +
def merge_final(date_batches):

    #Import existing final dataframe 
    bucket = 'methane-capstone' #S3 bucket location

    #Read current dataframes from s3
    subfolder = 'data/dt=latest'
    s3_path = bucket+'/'+subfolder 
    
    #grouped data
    file_name='data-zone-combined.parquet.gzip'
    data_location = 's3://{}/{}'.format(s3_path, file_name)
    current_df = pd.read_parquet(data_location)
    
    #non-grouped data
    file_name='data-address-combined.parquet.gzip'
    data_location = 's3://{}/{}'.format(s3_path, file_name)
    current_df_address = pd.read_parquet(data_location)
    
    
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    #Import pre-processed new data created recently from previous steps in this notebook
    
    start = date_batches[0]
    end = date_batches[-1]
    subfolder = 'data/pipeline-raw-data'
    s3_path = bucket+'/'+subfolder 
    
    #grouped data
    file_name=f'merged_{start}_{end}_data-zone-combined.parquet.gzip'  #need to update, currently anmed the other way
#     file_name=f'{start}_{end}_combined_data.parquet.gzip'
    new_df = pd.read_parquet('s3://{}/{}'.format(s3_path, file_name))   
    new_df['time_utc'] = pd.to_datetime(new_df['time_utc'])
    
    #non-grouped data
    file_name=f'merged_{start}_{end}_data-address-combined.parquet.gzip'  #need to update, currently anmed the other way
#     file_name=f'{start}_{end}_combined_data.parquet.gzip'
    new_df_address = pd.read_parquet('s3://{}/{}'.format(s3_path, file_name))   
    new_df_address['time_utc'] = pd.to_datetime(new_df_address['time_utc'])
   
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    #Check if all of the weather data columns are NaN's, if so, drop those rows (b/c ERA5 weather data hasn't been updated yet on Amazon Public Data Registry).

    #new_df.columns[7:21] are weather data from ERA5. If *all* of these are NaN, then dropna with how='all' will drop these rows.
    new_df = new_df[new_df['time_utc'] >= start].dropna(subset=new_df.columns[7:21], how='all')   #Drop NA's when all weather is missing
    new_df_address = new_df_address[new_df_address['time_utc'] >= start].dropna(subset=new_df_address.columns[16:30], how='all')   #Drop NA's when all weather is missing

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    #Merge two dataframes.

    assert sum(current_df.columns == new_df.columns) == len(current_df.columns)  #confirm that two dataframes have the same columns
    assert sum(current_df_address.columns == new_df_address.columns) == len(current_df_address.columns)  #confirm that two dataframes have the same columns
    # sum(df1.columns==df.2columns) tracks how many columns are the same between the two df
    # len(df1.columns) trakcs total number of columns.  
    # If the two sums are equal, all the columns are the same for df1 and df2.

    final_merged_df = current_df.append(new_df)
    final_merged_df['time_utc'] = pd.to_datetime(final_merged_df['time_utc'])
    final_merged_df.drop_duplicates(inplace=True)
    
    final_merged_df_address = current_df_address.append(new_df_address)
    final_merged_df_address['time_utc'] = pd.to_datetime(final_merged_df_address['time_utc'])
    final_merged_df_address.drop_duplicates(inplace=True)

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    # Save to S3 (replace old final dataframe)
    subfolder = 'data/dt=latest'
    s3_path = bucket+'/'+subfolder 

    file_name='data-zone-combined.parquet.gzip'
    final_merged_df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')
    data_location_csv_zone = 's3://methane-capstone/public_data/zone-data.csv'
    final_merged_df.to_csv(data_location_csv_zone) #publicly available dataset   
    
    file_name='data-address-combined.parquet.gzip'
    final_merged_df_address.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')    
    data_location_csv_address = 's3://methane-capstone/public_data/full-raw-data.csv'
    final_merged_df.to_csv(data_location_csv_address) #publicly available dataset       
    
    # Save to S3 (in archive folder for backup)
#     date=str(localtime.tm_year)+str(localtime.tm_mon)+str(localtime.tm_mday) #define date (for naming backup)
    subfolder = f'data/dt=archive/{start}_{end}'
    s3_path = bucket+'/'+subfolder
    
    file_name=f'{end}_data-zone-combined.parquet.gzip'
    final_merged_df.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')

    file_name=f'{end}_data-address-combined.parquet.gzip'
    final_merged_df_address.to_parquet('s3://{}/{}'.format(s3_path, file_name), compression='gzip')


# -

# # RUN PIPELINE!

# +
start=time.time()

show_time = True

#Run S5P Methane Data Extraction
start_s5p_extraction(date_batches)
methane_time=time.time()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
if show_time:
    print("Time to Finish Methane Data Extraction: ", methane_time-start)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#Run ERA5 Weather Data Extraction
start_era5_extraction(date_batches)
weather_time=time.time()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
if show_time:
    print("Time to Finish Weather Data Extraction: ", weather_time-methane_time)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#Run CA Climate Zone Data Addition
start_CACZ_addition(date_batches)
cacz_time=time.time()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
if show_time:
    print("Time to Finish CA ClimateZone Data Addition: ", cacz_time-weather_time)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#Run Data Pre Process to Group and Merge Data in Date Range
data_pre_process(date_batches)
preprocess_time=time.time()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
if show_time:
    print("Time to Pre Processing: ", preprocess_time-cacz_time)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

#Run Merging of New Dataframe to Old Dataframe
merge_final(date_batches)
merge_final_time=time.time()
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
if show_time:
    print("Time to Merge Final: ", merge_final_time-preprocess_time)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

print("Total Time: ", time.time()-start)
# -

5+5

# ## Compare new data with old data

# +
# bucket = 'methane-capstone'
# subfolder = 'data/pipeline-raw-data'
# s3_path = bucket+'/'+subfolder 
# file_name='2021-10-16_2021-10-31_combined_data.parquet.gzip'
# data_location = 's3://{}/{}'.format(s3_path, file_name)
# new_dataset = pd.read_parquet(data_location)
# new_dataset['time_utc'] = pd.to_datetime(new_dataset['time_utc'])

# +
# bucket = 'methane-capstone'
# subfolder = 'data/combined-raw-data'
# s3_path = bucket+'/'+subfolder 
# file_name='data-zone-combined.parquet.gzip'
# data_location = 's3://{}/{}'.format(s3_path, file_name)
# current_dataset = pd.read_parquet(data_location)
# current_dataset['time_utc'] = pd.to_datetime(current_dataset['time_utc'])

# +
# day="2021-10-16"
# new_dataset[new_dataset['time_utc']== day].sort_values(by='BZone')

# +
# current_dataset[current_dataset['time_utc']== day].sort_values(by='BZone').head()
# -




