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
# * `.parquet` format

# ### Install steps
#
# You may need to do the following pip installs
#
# 1. `!pip install xarray`
# 2. `!pip install geopandas`
# 3. `!pip install shapely`
# 4. `!pip install netCDF4`
#
#

# !pip install geopandas

# ### TODOS:
# * Check speed on a couple different machines

# +

import traceback
import sys
import subprocess
import pickle
import os
import glob

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

# ### Load in CA GeoJSON --> Polygon
# * Check point in Polygon - https://stackoverflow.com/questions/43892459/check-if-geo-point-is-inside-or-outside-of-polygon
# * Link to GeoJSON - https://github.com/ropensci/geojsonio/blob/master/inst/examples/california.geojson
#
# We will use this to strictly filter data only for California

ca_geo_path = "/root/methane/data_processing/resources/california.geojson"
ca_geo_file = open(ca_geo_path)
ca_gpd = gpd.read_file(ca_geo_file)
cali_polygon = ca_gpd['geometry'][0]
print(ca_gpd)
cali_polygon



# ### Sanity Check 
#
# Do a quick check to ensure that the point checking is working 

# +
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



# ### Get FULL list of CA orbit scanlines
#
# This comes from Jaclyn's selections

# +
#Jaclyn's File
with open('/root/methane/data_processing/data_maps/ca_filenames.pickle', 'rb') as handle:
    ca_files = pickle.load(handle)    
print("Number of scan files:", len(ca_files))

#AWS OFFL Data Catalog
with open('/root/methane/data_processing/data_maps/data_catalog.pickle', 'rb') as handle:
    data_catalog = pickle.load(handle)
    
print("Number of years:", len(data_catalog))
# -

# ### Check File Types

# +
file_types = []
for fn_str in ca_files:
    file_type = fn_str.split("_")[1]
    file_types.append(file_type)

Counter(file_types)
# -



# ### Get processing window --> `cur_file_window`
#
# Specify
# * `start_dt` - earliest can be 11/28/2018
# * `end_dt`- latest can be 10/01/2021, but....lets stick to 09/30/2021
#
# #### Would be best to do 4 different runs of the below code cell, one for each year

# +
start_dt = datetime(2018, 1, 1).strftime('%Y%m%d')
end_dt = datetime(2018, 12, 31).strftime('%Y%m%d')
batch_size = 13

cur_file_window = []
non_offl_files = []

#Ensure this is true!
assert start_dt <= end_dt

def getBatches(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
#Form Batches
for ca_file in ca_files:
    collection_date = ca_file.split("____")[1].split("T")[0]
    if collection_date >= start_dt and collection_date <= end_dt:

        #Only look at offline files...? 
        if "OFFL" in ca_file:
            cur_file_window.append(ca_file)

        else:
            non_offl_files.append(ca_file)

cur_file_window = sorted(cur_file_window, reverse=False) #oldest --> newest
file_batches = list(getBatches(cur_file_window, batch_size))


print("Number of files: ", len(cur_file_window))
if len(non_offl_files) > 0: print("Number of files, can NOT use: ", len(non_offl_files), '\n')
print("Number of batches: ", len(file_batches))
print("Number of files in batch: ~", batch_size)

# -



# ### Batch Breakdown
# * batch size = 13 
#
# #### 2018
# * 58 NC files, 5 batches => 5 parquet files
#
# #### 2019
# * 635 NC files, 49 batches => 49 parquet files
#
# #### 2020
# * 636 NC files, 49 batches => 49 parquet files
#
# #### 2021
# * 473 NC files, 37 batches => 37 parquet files
#



# ### Pipeline Steps
# <em> All files in `cur_file_window` will be processed.<em>
#
# **NOTE**
# For every batch we will write a file. Therefore we will **NOT** have a month wise split for data right now. In a post processign step we will aggregate by month for ease of access. 
#
# Steps:
# 1. Define the date window 
# 2. Setup file batches to download
# 3. Download whole batch to local
# 4. Parse each file in batch, and build a dataframe for each batch
# 5. For every batch we will flush the data to a `.parquet.gzip` straight to S3 in respective folder.
#     * Format `startdt_enddt_meth.parquet.gzip`
# 6. Delete contents inside Sagemaker batch folder where `.nc` files were downloaded for a respective batch
# 7. Write out a `.txt` file that writes out all the steps that happened in the pipeline



# +
#### CONFIG ####
pipeline_start = False

local_pipe_log_path = '/root/methane/data_processing/pipeline_runs/'
local_path = '/root/methane/data_processing/nc_data/cur_batch/'
bucket = 'methane-capstone'
subfolder = 'batch-raw-data'
s3_path = bucket+'/'+subfolder

delete_local_files = True
cur_file_window_set = set(cur_file_window)
qa_thresh = 0.0


#### HELPERS ####
def getNCFile(batch_file_name, batch_file_num, local_path, outF):
    '''Function to get a S5P file from AWS S3 to local'''
    start = time.time()
    try:
        collection_date = batch_file_name.split("____")[1].split("T")[0]
        year = collection_date[:4]
        month = collection_date[4:6]
        day = collection_date[6:]

        command = ['aws','s3','cp', 
                   f's3://meeo-s5p/OFFL/L2__CH4___/{year}/{month}/{day}/{batch_file_name}',
                   local_path]
        print_write(f"\tbatch_file_num: {batch_file_num} " + " ".join(command), outF)
        subprocess.check_output(command)
    except:
        print_write(f"ISSUE GETTING: {batch_file_name}", outF)
    end = time.time()


def getInputFiles(local_path):
    '''Get list of input files'''
    input_files = sorted(list(iglob(join(local_path, '**', '*CH4*.nc' ), recursive=True)))
    return input_files

def loadNCFiles(input_files, outF):
    '''Use xarray to load .nc file S5p products'''    
    start = time.time()
    s5p_products = {}
    for file_name in input_files:
        name = file_name.split('/')[-1]
        if name in cur_file_window_set:
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

def filter_in_cali(row):
    '''Filter apply function for CA'''
    point = Point(row['lon'], row['lat'])
    return cali_polygon.contains(point)

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




###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 
###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### ###### 

if pipeline_start:
    
    #Give the pipeline log file a name
    pipe_start = cur_file_window[0].split("____")[1].split("T")[0]
    pipe_end = cur_file_window[-1].split("____")[1].split("T")[0]    
    
    outF = open(f"{local_pipe_log_path}pipe_log_{pipe_start}_{pipe_end}.txt", "w")
    
    init = time.time()
    print_write("######### PIPELINE STARTED #########", outF)
    print_write(f"Total Files: {len(cur_file_window)}", outF)
    print_write(f"Total Batches: {len(file_batches)}", outF)
    print_write(f"Batch size: ~ {batch_size}", outF)
    
    try:

        for batch_num, batch in enumerate(file_batches, 1):

            print_write(f"\nbatch_num: {batch_num} - start download", outF)
            batch_dn_start = time.time()
            for batch_file_num, batch_file in enumerate(batch, 1):

                #Download one file
                getNCFile(batch_file, batch_file_num, local_path, outF)
                #continue

            batch_dn_end = time.time()
            print_write(f"batch_num: {batch_num} - finish download, time_taken: {getHumanTime(batch_dn_end-batch_dn_start)}", outF)

            #Get list of input files
            input_files = getInputFiles(local_path)

            #Load NC Files into dictionary
            print_write(f"batch_num: {batch_num} - start NC files load", outF)
            s5p_products, load_nc_time = loadNCFiles(input_files, outF)
            print_write(f"batch_num: {batch_num} - end NC files load, time_taken: {load_nc_time}", outF)

            #One entire batch is processsed into `cur_df`
            cur_batch_df, process_time = processFiles(s5p_products, qa_thresh, outF)

            ### Write `cur_batch_df` for each batch to S3 ###
            try:

                first_day = batch[0].split("____")[1].split("T")[0]
                last_day = batch[-1].split("____")[1].split("T")[0]
                year = first_day[:4]

                file_name=f'{first_day}_{last_day}_meth.parquet.gzip'
                cur_batch_df.to_parquet('s3://{}/{}'.format(s3_path+f'/{year}', file_name), compression='gzip')

            except:
                write_loc = 's3://{}/{}'.format(s3_path+f'/{year}', file_name)
                print_write(f"ERROR S3 WRITE: {write_loc}", outF)

            if delete_local_files:
                for f in input_files:
                    os.remove(f)
                print_write("deleted nc files", outF)

            batch_end_time = time.time()

            print_write(f"batch_num: {batch_num} batch_total_time: {getHumanTime(batch_end_time - batch_dn_start)} download_time: {getHumanTime(batch_dn_end-batch_dn_start)}, file_count: {len(batch)}, load_nc_time: {load_nc_time}, process_time: {process_time}, cur_batch_df_shape: {cur_batch_df.shape}", outF)
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
