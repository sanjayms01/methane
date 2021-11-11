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

import numpy as np
import pandas as pd
import time 
from datetime import datetime
import random
from random import randrange
from datetime import timedelta
import os
import glob
import subprocess
import re
import traceback
from tqdm.auto import tqdm

# +
#### CONFIG ####

location_name = 'testsite3'

pipeline_start = True
bucket = 'methane-capstone'
subfolder = 'month-raw-data'
s3_path = bucket+f'/data/{location_name}/'+subfolder
s3_batch_raw_path = bucket+f'/data/{location_name}/batch-raw-data'

years = ['2018', '2019', '2020','2021']
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

if pipeline_start:
    
    debut = time.time()
    print("######### MONTHLY PIPELINE STARTED #########")
    
    for year in tqdm(years):

        # All files ending with .parquet.gzip
        file_names = subprocess.check_output(['aws','s3','ls', f's3://{s3_batch_raw_path}/{year}/'])
        vals = [entry.decode('utf-8') for entry in file_names.split()]
        zip_file_names = [x for x in vals if '.gzip' in x]

        #print(f"year: {year}, num_files: {len(zip_file_names)}")

        #Load 1 year worth of data
        start_load = time.time()
        df_year = pd.DataFrame(columns=columns, dtype='object')
        for file_name in tqdm(zip_file_names, desc='file loading', leave=False):
            df_cur_batch = pd.read_parquet(f's3://{s3_batch_raw_path}/{year}/{file_name}')
            #print(f"\t file_name: {file_name}, shape: {df_cur_batch.shape}")
            df_year = pd.concat([df_year, df_cur_batch], ignore_index=True)

        #Add `year-month` column
        df_year['time_utc'] = pd.to_datetime(df_year['time_utc'], infer_datetime_format=True)
        df_year.insert(1, 'year_month', df_year['time_utc'].dt.to_period('M').astype(str))
        end_load = time.time()
        
        #print(f"\t load_time: {end_load - start_load}, shape: {df_year.shape}, month_coverage: {df_year['year_month'].nunique()}")
        #print()
        
        # For every `year-month`
        for period in tqdm(sorted(df_year['year_month'].unique()), desc='writing file'):
            df_year_month = df_year[df_year['year_month'] == period]
            df_year_month = df_year_month[df_year_month['qa_val'] >= 0.5]
            df_year_month = df_year_month.sort_values(by=['time_utc'])
            new_file_name = f'{str(period)}' + '-meth.parquet.gzip'
            s3_file_path = 's3://{}/{}'.format(s3_path+f'/{year}', new_file_name)
            #print(f"\t\t writing to S3: {s3_file_path}")
            df_year_month.to_parquet(s3_file_path, compression='gzip', coerce_timestamps="us")
        print()
    fin = time.time()
    print(f"MONTHLY PIPELINE DONE, time_taken:{fin-debut}")
else:
    print("MONTHLY PIPELINE CLOSED")
    
# -





# ### Another pass at combining the data to validate the size

# +
data_key_list_2018 = [ '2018-11-meth.parquet.gzip',  '2018-12-meth.parquet.gzip']

data_key_list_2019 = [ '2019-01-meth.parquet.gzip',  '2019-02-meth.parquet.gzip', '2019-03-meth.parquet.gzip', '2019-04-meth.parquet.gzip', '2019-05-meth.parquet.gzip',
 '2019-06-meth.parquet.gzip', '2019-07-meth.parquet.gzip', '2019-08-meth.parquet.gzip', '2019-09-meth.parquet.gzip', '2019-10-meth.parquet.gzip', '2019-11-meth.parquet.gzip', '2019-12-meth.parquet.gzip']

data_key_list_2020 = [ '2020-01-meth.parquet.gzip', '2020-02-meth.parquet.gzip', '2020-03-meth.parquet.gzip', '2020-04-meth.parquet.gzip', '2020-05-meth.parquet.gzip',
 '2020-06-meth.parquet.gzip', '2020-07-meth.parquet.gzip', '2020-08-meth.parquet.gzip', '2020-09-meth.parquet.gzip', '2020-10-meth.parquet.gzip', '2020-11-meth.parquet.gzip', '2020-12-meth.parquet.gzip']

data_key_list_2021 = [ '2021-01-meth.parquet.gzip', '2021-02-meth.parquet.gzip', '2021-03-meth.parquet.gzip', '2021-04-meth.parquet.gzip', '2021-05-meth.parquet.gzip',
 '2021-06-meth.parquet.gzip', '2021-07-meth.parquet.gzip', '2021-08-meth.parquet.gzip', '2021-09-meth.parquet.gzip', '2021-10-meth.parquet.gzip']


bucket = 'methane-capstone'
subfolder = 'month-raw-data'

def read_sentinel5p(year, filename):
    s3_path = bucket+f'/data/{location_name}/'+subfolder + '/' + year
    data_location = 's3://{}/{}'.format(s3_path, filename)
    return data_location

print("Check that we have all the months we want")
print(len(data_key_list_2018))
print(len(data_key_list_2019))
print(len(data_key_list_2020))
print(len(data_key_list_2021))

# +
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

df_all = pd.DataFrame(columns=columns, dtype='object')

for year_list_ind, year in enumerate(['2018','2019','2020','2021'], 0):
    
    month_list = []
    if year_list_ind == 0:
        month_list = data_key_list_2018
    elif year_list_ind == 1:
        month_list = data_key_list_2019
    elif year_list_ind == 2:
        month_list = data_key_list_2020
    elif year_list_ind == 3:
        month_list = data_key_list_2021
    else:
        print("WE GOT ISSUES")

    df_cur_year = pd.DataFrame(columns=columns, dtype='object')
    for filename in month_list:
        print(read_sentinel5p(year, filename))
        try:
            df_cur_month = pd.read_parquet(read_sentinel5p(year, filename))
            df_cur_year = pd.concat([df_cur_year, df_cur_month], ignore_index=True)    
            print(f"month: {filename}, shape: {df_cur_month.shape}")
        except Exception:
            #traceback.print_exc()
            print("\tPROBLEM FILE:",filename)

    print(f"Year: {year}, shape: {df_cur_year.shape}")
    df_all = pd.concat([df_all, df_cur_year], ignore_index=True)
    

print(f"Total shape: {df_all.shape}")
# -

gzip_write_path = f's3://methane-capstone/data/{location_name}/combined-raw-data/combined-raw-2.parquet.gzip'
parquet_write_path = f's3://methane-capstone/data/{location_name}/combined-raw-data/combined-raw-2.parquet'

df_all.to_parquet(parquet_write_path, coerce_timestamps="us")

df_all.to_parquet(gzip_write_path, compression='gzip', coerce_timestamps="us")

#Cross checking John's write
df_check = pd.read_parquet(f's3://methane-capstone/data/{location_name}/combined-raw-data/combined-raw-2.parquet')
print(df_check.shape)

# ### Checks out perfect! 

# !pwd

# !jupytext --to py month_processing.ipynb


