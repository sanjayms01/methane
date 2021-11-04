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
# This notebook has the full pipeline to parse Sentinel 5p data from AWS s3.
#
#
# * Partition will be by **time**
# * `.parquet.gzip` format

# # Install steps

# +
# # # You may need to do the following pip installs
# # !pip install xarray
# # !pip install geopandas
# # !pip install shapely
# # !pip install netCDF4
# -



# # Imports

# +
import traceback
import sys
import subprocess
import pickle
import os
import glob
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
# -



# # Create Filter for California Data
# We will use this to strictly filter data only for California 
#
# 1) Load in CA GeoJSON --> Polygon
# * Check point in Polygon - https://stackoverflow.com/questions/43892459/check-if-geo-point-is-inside-or-outside-of-polygon
# * Link to GeoJSON - https://github.com/ropensci/geojsonio/blob/master/inst/examples/california.geojson

ca_geo_path = "/root/methane/data_processing/resources/california.geojson"
ca_geo_file = open(ca_geo_path)
ca_gpd = gpd.read_file(ca_geo_file)
cali_polygon = ca_gpd['geometry'][0]
print(ca_gpd)
cali_polygon


# +
### Sanity Check 
#Do a quick check to ensure that the point checking is working 

def inCalifornia(lat,lon):
    '''Given lat, lon. Return Boolean if in California'''
    point = Point(lon, lat)
    return cali_polygon.contains(point)

#Point in Arizona
az_y= 32.866806
az_x = -114.35925
print("In California") if inCalifornia(az_y, az_x) else print("NOT in California")

#Point in California
ca_y = 37.962030
ca_x = -121.186863
print("In California") if inCalifornia(ca_y, ca_x) else print("NOT in California")


#Point in Lake Tahoe, CA (border)
ta_y = 38.913072
ta_x = -119.913452
print("In California") if inCalifornia(ta_y, ta_x) else print("NOT in California")

#Point in Carson City, NV (border)
cars_y = 39.155575
cars_x = -119.721257
print("In California") if inCalifornia(cars_y, cars_x) else print("NOT in California")
# -



# # Create Date Range to Extract S5P Methane Data
# Each date would be a batch of 14 orbit path files

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

print(date_batches)


# -

# ### Pipeline Steps
# <em> All files with dates in `date_batches` will be processed.<em>
#
# **NOTE**
# For every batch we will write a file. Each batch is one day so it would be a file for each date. Then we concatentate all dates to the main dataframe.
#
# Steps:
# 1. We have already defined a date range window.  
# 2. Download whole batch to local.  
# 3. Parse each file in batch, and build a dataframe for each batch
# 4. For every batch we will flush the data to a `.parquet.gzip` straight to S3 in respective folder.
#     * Format `date_meth.parquet.gzip`
# 6. Delete contents inside Sagemaker batch folder where `.nc` files were downloaded for a respective batch
# 7. Write out a `.txt` file that writes out all the steps that happened in the pipeline

# +
#### HELPERS ####
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
        # print_write(f"\tbatch_file_num: {batch_file_num} " + " ".join(command), outF)
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
    input_files = sorted(list(iglob(join(local_path, '**', '*CH4*.nc' ), recursive=True)))
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
    for file_number, product_key in enumerate(s5p_products, 1):
        start_time = time.time()
        s5p_prod = s5p_products[product_key]
        df_cur_scan = pd.DataFrame(columns=columns)
        times_utc = np.array(s5p_prod.time_utc[0, :])

        print_write(f'\tfile_num: {file_number} - {product_key}. start_parse.', outF)
        for index, ts in enumerate(times_utc, 0):
            lats = np.array(s5p_prod.latitude[0, :, :][index])
            lons = np.array(s5p_prod.longitude[0, :, :][index])
            qa_vals = np.nan_to_num(s5p_prod.qa_value[0, :, :][index], nan=-9999)
            methane_mixing_ratio = np.nan_to_num(s5p_prod.methane_mixing_ratio[0, :, :][index], nan=-9999)
            methane_mixing_ratio_precision = np.nan_to_num(s5p_prod.methane_mixing_ratio_precision[0, :, :][index], nan=-9999)    
            methane_mixing_ratio_bias_corrected = np.nan_to_num(s5p_prod.methane_mixing_ratio_bias_corrected[0, :, :][index], nan=-9999)

            cur_ts_dict = {
                'time_utc' : [ts] * len(lats),
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
    return df_ca, getHumanTime(end_time-init_start)


# +
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

if pipeline_start:
    
   
    #Give the pipeline log file a name
    pipe_start = str(date)   
    outF = open(f"{local_pipe_log_path}pipe_log_{pipe_start}.txt", "w")
    print_write("######### PIPELINE STARTED #########", outF)
   
    init = time.time()    
    
    try:

        for batch_num, date in enumerate(date_batches, 1):

            print_write(f"\nbatch_num: {batch_num} - start download", outF)
            batch_dn_start = time.time()
            
            #Download all 14 .NC files on that one date
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
    print("PIPELINE CLOSED")
# -

# ### Test Reading back data

# +
# data_key = '20181221_20181228_meth.parquet.gzip'
# data_location = 's3://{}/{}'.format(s3_path+'/2018', data_key)
# test_df = pd.read_parquet(data_location)
# print(test_df.shape)
# test_df.head()

# +
# # Testing Methane Mask
# #Methane Mask
# meth_ca_mask_df = (test_df.methane_mixing_ratio != -9999) & \
#                   (test_df.methane_mixing_ratio_precision != -9999) & \
#                   (test_df.methane_mixing_ratio_bias_corrected != -9999)
# #Join Masks
# df_2 = test_df[meth_ca_mask_df]
# print(df_2.shape)
# df_2.head()

# +
# Performance Tracker
# 2 vCPU + 4 GiB ---> 46.80413150787353
