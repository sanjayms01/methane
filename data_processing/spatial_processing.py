import sys
import getopt
import optparse
import glob
import time
# import numpy as np
# import pandas as pd
# import geojson
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
# import geopandas as gpd


def main():
    parser = optparse.OptionParser()

    parser.add_option('--region',
        action="store", dest="region", nargs=4,
        help="region")
    
    parser.add_option('--res',
        action="store", dest="res", nargs=3,
        help="resolution")
    
    parser.add_option('-q', '--quality',
        action="store", dest="quality",
        help="quality (>=)")
    
    parser.add_option('-i', '--interpolate',
        action="store", dest="interpolate", default='time'
        help="interpolate method (time, linear, spline)")

    options, args = parser.parse_args()
    print('CLI OPTIONS: ', options)
    
    # Load Data
    df = loadData(options)
    print('LOADING DATA END')
    print('----------------')
    
    # Preprocess Data
    df_processed = preprocessData(df, options)
    print('PREPROCESSING DATA END')
    print('----------------------')
    
    # Saving Data To S3
    writeToS3(df_processed)
    print('WRITING TO S3 END')
    print('-----------------')
    

def loadData(options):
    # {'region': ('-123', '37', '-121', '38'), 'res': None, 'quality': None}
    
    start = time.time()
    s3_file_path = 's3://methane-capstone/data/combined-raw-data/combined-raw.parquet.gzip'
    
    print('LOADING DATA')
    print('------------')
    
    df = pd.read_parquet(s3_file_path)
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    df['year_month'] = df.year_month.astype(str)
    print('Dataframe Shape: ', df.shape)
    end = time.time()
    
    print("Load Time: ", end-start)
    print()
    
    if options['all']:
        
    
    # Check CLI Options
    if options['region']:
        print('Using Region...')
        ll_lon, ll_lat, ur_lon, ur_lat = options['region']
        
        geo_mask = (df.lon > ll_lon) & \
               (df.lat > ll_lat) & \
               (df.lon < ur_lon) & \
               (df.lat < ur_lat)
        
        trim_cols = ['time_utc','lat','lon', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision' 'methane_mixing_ratio_bias_corrected']
        
        
    elif options['res']:
        print('Using Resolution...\n')
        res_val, rounded_lat, rounded_lon = options['res']
        print('SPOT Resolution: ', res_val)
        
        if res_val == 0.1:
        geo_mask = (df.rn_lat_1 == rounded_lat) & \
                   (df.rn_lon_1 == rounded_lon)
        trim_cols = ['time_utc','rn_lat_1','rn_lon_1', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']
        
    elif res_val == 0.2:
        geo_mask = (df.rn_lat_2 == rounded_lat) & \
                   (df.rn_lon_2 == rounded_lon)
        trim_cols = ['time_utc','rn_lat_2','rn_lon_2', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']

    elif res_val == 0.5:
        geo_mask = (df.rn_lat_5 == rounded_lat) & \
                   (df.rn_lon_5 == rounded_lon)
        trim_cols = ['time_utc','rn_lat_5','rn_lon_5', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']
        
    elif res_val == 1.0:
        geo_mask = (df.rn_lat == rounded_lat) & \
                   (df.rn_lon == rounded_lon)
        trim_cols = ['time_utc','rn_lat','rn_lon', 'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision', 'methane_mixing_ratio_bias_corrected']
        
    df_trim = df[geo_mask][trim_cols]
    print(f"Losing {'{:,}'.format(df.shape[0] - df_trim.shape[0])} rows")
    print()
    print(f"df_trim shape: {df_trim.shape}")
    print()
        
    if options['quality']:
        #Filter df by quality
        print("Filtering dataframe by quality...")
        df_filtered = df_trim[df_trim.qa_val >= options['quality']]
        print('df_filtered shape: ', df_filtered.shape)
        print()
        return df_filtered
        
    
    return df_trim


def preprocessData(df_trim, options):
    # ## Pre-Process Data
    # * one reading per day
    # * understand missing days
    #
    #
    # #### Interpolation Methods:
    # * https://towardsdatascience.com/how-to-interpolate-time-series-data-in-apache-spark-and-python-pandas-part-1-pandas-cff54d76a2ea
    # * https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
    #
    # #### Upsampling to Hourly Data:
    # * https://kanoki.org/2020/04/14/resample-and-interpolate-time-series-data/
    #
    #
    # Documentation:
    # * https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
    #
    #
    # * ffill - Forward Fill
    # * bfill - Backward Fill
    # * Interpolation
    #
    # I chose to stick with `time` interpolation!
    #
    #
    
    print('Preprocessing Data')
    print('------------------')
    
    #Group by date. one average reading per day
    df_trim_day = df_trim.groupby(df_trim.time_utc.dt.date).agg({'methane_mixing_ratio_bias_corrected': 'mean',
                                                             'methane_mixing_ratio': 'mean',
                                                             'methane_mixing_ratio_precision': 'mean',
                                                            }).reset_index()
    
    
    #To also get the number of readings we have each day
    df_trim_reading_count = df_trim.groupby(df_trim.time_utc.dt.date).agg({
                                                                 'methane_mixing_ratio': 'count',
                                                                }).reset_index().rename(columns={"methane_mixing_ratio": "reading_count"})

    #Join this onto our data set
    df_trim_day = df_trim_day.merge(df_trim_reading_count, 'inner')

    #Add in missing date rows
    df_trim_day = df_trim_day.set_index('time_utc').asfreq('D')
    
    print('Interpolating Data...')
    
    #Interpolate Data
    method = options['interpolate']
    print('Method: ', method)
    order = 3

    df_interpol = df_trim_day.resample('D').mean()
    df_trim_day['mmrbc_i'] = df_interpol['methane_mixing_ratio_bias_corrected'].interpolate(method=method) #, order=order)
    df_trim_day['mmr_i'] = df_interpol['methane_mixing_ratio'].interpolate(method=method) #, order=order)
    df_trim_day['mmrp_i'] = df_interpol['methane_mixing_ratio_precision'].interpolate(method=method) #, order=order)

    #Set any interpolated date to `0`
    df_trim_day['reading_count'] = df_trim_day['reading_count'].fillna(value=0)

    df_trim_day = df_trim_day.reset_index()
    print('Interpolated Data Shape: ', df_trim_day.shape)
    print()
    
    return df_trim_day

def writeToS3(df):
    bucket = 'methane-capstone'
    subfolder = 'batch-raw-data'
    s3_path = bucket+'/'+subfolder
    
    print('WRITING TO S3')
    print('-------------')
    
    try:
        file_name=f'{}_{}.parquet.gzip' # CHANGE NAME!
        df.to_parquet('s3://{}/{}'.format(s3_path+'/', file_name), compression='gzip')

    except:
#         write_loc = 's3://{}/{}'.format(s3_path+'/', file_name)
#         print_write(f"ERROR S3 WRITE: {write_loc}", outF)


# Call it processed_data


if __name__ == '__main__':
    main()