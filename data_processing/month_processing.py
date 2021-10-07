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

# +
#### CONFIG ####
pipeline_start = False
bucket = 'methane-capstone'
subfolder = 'month-raw-data'
s3_path = bucket+'/'+subfolder
s3_batch_raw_path = bucket+'/batch-raw-data'

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
    
    for year in years:

        # All files ending with .parquet.gzip
        file_names = subprocess.check_output(['aws','s3','ls', f's3://{s3_batch_raw_path}/{year}/'])
        vals = [entry.decode('utf-8') for entry in file_names.split()]
        zip_file_names = [x for x in vals if '.gzip' in x]

        print(f"year: {year}, num_files: {len(zip_file_names)}")

        #Load 1 year worth of data
        start_load = time.time()
        df_year = pd.DataFrame(columns=columns)
        for file_name in zip_file_names:
            df_cur_batch = pd.read_parquet(f's3://{s3_batch_raw_path}/{year}/{file_name}')
            print(f"\t file_name: {file_name}, shape: {df_cur_batch.shape}")
            df_year = pd.concat([df_year, df_cur_batch], ignore_index=True)

        #Add `year-month` column
        df_year['time_utc'] = pd.to_datetime(df_year['time_utc'], infer_datetime_format=True)
        df_year.insert(1, 'year_month', df_year['time_utc'].dt.to_period('M'))
        end_load = time.time()
        
        print(f"\t load_time: {end_load - start_load}, shape: {df_year.shape}, month_coverage: {df_year['year_month'].nunique()}")
        print()
        
        # For every `year-month`
        for period in sorted(df_year['year_month'].unique()):
            df_year_month = df_year[df_year['year_month'] == period]
            df_year_month = df_year_month.sort_values(by=['time_utc'])
            new_file_name = f'{str(period)}' + '-meth.parquet.gzip'
            s3_file_path = 's3://{}/{}'.format(s3_path+f'/{year}', new_file_name)
            print(f"\t\t writing to S3: {s3_file_path}")
            df_year_month.to_parquet(s3_file_path, compression='gzip')
        print()
    fin = time.time()
    print(f"MONTHLY PIPELINE DONE, time_taken:{fin-debut}")
else:
    print("MONTHLY PIPELINE CLOSED")
    
# -






