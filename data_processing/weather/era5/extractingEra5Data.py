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

# # ERA5 DATA EXTRACTION (FOR CALIFORNIA DATA)
#
# Step 1: Extract all unique latitude/longitude combinations from Methane Dataframe. (Used to extract ERA5 data).  
# Step 2: Inspect AWS ERA5 directory and store relevant weather variable names.  
# Step 3a: Download raw ERA5 data and write to local directory.  
# Step 3b: Extract only weather from California and write to local directory.

# +
# # !pip install netcdf4
# # !pip install h5netcdf
# # REMEMBER TO RESTART KERNEL
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
import time

# +
#IMPORT METHANE DATA AND ONLY KEEP THE LAT/LON COMBINATIONS TO EXTRACT FROM ERA5 DATA

# Read in Methane Data
bucket = 'methane-capstone'
subfolder = 'data/combined-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='combined-raw.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)
methane_df = pd.read_parquet(data_location)

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

# +
# date_list = [datetime.date(2021, 9, 1),  #done
#              datetime.date(2021, 8, 1),  #done
#              datetime.date(2021, 7, 1),  #done
#              datetime.date(2021, 6, 1),  #done
#              datetime.date(2021, 5, 1),  #done
#              datetime.date(2021, 4, 1),  #done
#              datetime.date(2021, 3, 1),  #done
#              datetime.date(2021, 2, 1),  #done
#              datetime.date(2021, 1, 1),  #done
#                  datetime.date(2020, 12, 1),  #done
#                  datetime.date(2020, 11, 1),
#                  datetime.date(2020, 10, 1),
#                  datetime.date(2020, 9, 1),
#                  datetime.date(2020, 8, 1),
#                  datetime.date(2020, 7, 1),
#                  datetime.date(2020, 6, 1),
#                  datetime.date(2020, 5, 1),
#                  datetime.date(2020, 4, 1),
#                  datetime.date(2020, 3, 1),
#                  datetime.date(2020, 2, 1),
#                  datetime.date(2020, 1, 1),
#                      datetime.date(2019, 12, 1),
#                      datetime.date(2019, 11, 1),
#                      datetime.date(2019, 10, 1),
#                      datetime.date(2019, 9, 1),
#                      datetime.date(2019, 8, 1),
#                      datetime.date(2019, 7, 1),
#                      datetime.date(2019, 6, 1),
#                      datetime.date(2019, 5, 1),
#                      datetime.date(2019, 4, 1),
#                      datetime.date(2019, 3, 1),
#                      datetime.date(2019, 2, 1),
#                      datetime.date(2019, 1, 1),
#                          datetime.date(2018, 12, 1),
#                          datetime.date(2018, 11, 1)]

date_list = [    datetime.date(2020, 11, 1),
                 datetime.date(2020, 10, 1),
                 datetime.date(2020, 9, 1),
                 datetime.date(2020, 8, 1),
                 datetime.date(2020, 7, 1),
                 datetime.date(2020, 6, 1),
                 datetime.date(2020, 5, 1),
                 datetime.date(2020, 4, 1),
                 datetime.date(2020, 3, 1),
                 datetime.date(2020, 2, 1),
                 datetime.date(2020, 1, 1),
                     datetime.date(2019, 12, 1),
                     datetime.date(2019, 11, 1),
                     datetime.date(2019, 10, 1),
                     datetime.date(2019, 9, 1),
                     datetime.date(2019, 8, 1),
                     datetime.date(2019, 7, 1),
                     datetime.date(2019, 6, 1),
                     datetime.date(2019, 5, 1),
                     datetime.date(2019, 4, 1),
                     datetime.date(2019, 3, 1),
                     datetime.date(2019, 2, 1),
                     datetime.date(2019, 1, 1),
                         datetime.date(2018, 12, 1),
                         datetime.date(2018, 11, 1)]

# date_list = [datetime.date(2020,11,1)]

# +
# date = datetime.date(2020,12,1) # UPDATE TO DESIRED DATE*************************************

for date in date_list:
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
        data_file_ptrn = 'raw_data/{year}{month}_{var}.nc'

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
        df.to_parquet('data/{}_{}.parquet'.format(date.strftime('%Y_%m'),var))
        print("time to extract data (seconds): ", time.time()-start2)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("time to download all variables (secs): ", time.time()-start)
# -

# READ FILE TO TEST THAT IT WORKED
var='2021_08_air_temperature_at_2_metres'
file_location = 'data/'
file_name= var + '.parquet'
test_df = pd.read_parquet(file_location+file_name)
test_df













# +
# # MISC FUNCTIONS FOR CONVERTING DATA

# # Temperature is in Kelvin.
# def kelvin_to_celcius(t):
#     return t - 273.15

# def kelvin_to_fahrenheit(t):
#     return t * 9/5 - 459.67

# # ds_locs_f = ds_locs.apply(kelvin_to_fahrenheit)
# # df_f = ds_locs_f.to_dataframe()
# # df_f.describe()

# +
# #TEST WHICH GRANULARITY (scratch work)
# value=1
# print(methane_df['rn_lat_1'][value])
# print(methane_df['rn_lat_2'][value])
# print(methane_df['rn_lat_5'][value])
# print(methane_df['rn_lat'][value])

# lat_lon_columns =[['lat', 'lon'],
#                   ['rn_lat_1', 'rn_lon_1'],
#                   ['rn_lat_2', 'rn_lon_2'],
#                   ['rn_lat_5', 'rn_lon_5'],
#                   ['rn_lat', 'rn_lon']]

# for l in lat_lon_columns:
#     lat_lon = methane_df.groupby(l).count().reset_index()
#     print(l, ' combination count: ', lat_lon.shape[0])
# #     lat_lon.head(3)

# # 10x15 = 150seconds for 63 combos     /  2.44mb / 88mb
# # 211 combos -> ~ x3.4   ->  450?             /  8.3mb  /300mb
# # 964 combos -> ~ x15.5  ->   2325, 40 mins.  /  38mb  / 1368mb, 1.3gb
# # 3070 combos -> ~ x50


# +
# #Create list of lat/lon combinations
# lat_lon = methane_df.groupby(['rn_lat_2', 'rn_lon_2']).count().reset_index()
# lat_lon_combo = []
# for lat,lon in zip(list(lat_lon['rn_lat_2']), list(lat_lon['rn_lon_2'])):
#         lat_lon_combo.append([lat,lon])
# print('length:', len(lat_lon_combo))
# lat_lon_combo[:5]

# # location coordinates (to look up in ERA5 weather data and save)
# locs = [ ]
    
# for combo in lat_lon_combo:
#     combined = {}
#     combined['name'] = str(combo[0])+str(combo[1])
#     combined['lon'] = combo[1]
#     combined['lat'] = combo[0]
#     locs.append(combined)

# # convert westward longitudes to degrees east,  ERA5 data uses a 0-360 system instead of -180 to 180
# for l in locs:
#     if l['lon'] < 0:
#         l['lon'] = 360 + l['lon']
# locs[:5]

# +
# #WRITE DATAFRAME FOR EACH WEATHER VARIABLE AT SPECIFIED LAT/LON COMBOS FOR ONE MONTH

# start=time.time()
# # select date and variable of interest
# # date = datetime.date(2021,9,1)  #ALREADY SPECIFIED ABOVE

# for var in var_name_list:
    
#     start2=time.time()
#     var=str(var)
    
#     #DOWNLOAD DATA FROM AWS BUCKET
    
#     # file path patterns for remote S3 objects and corresponding local file
#     s3_data_ptrn = '{year}/{month}/data/{var}.nc'
#     data_file_ptrn = 'raw_data/{year}{month}_{var}.nc'

#     year = date.strftime('%Y')
#     month = date.strftime('%m')
#     s3_data_key = s3_data_ptrn.format(year=year, month=month, var=var)
#     data_file = data_file_ptrn.format(year=year, month=month, var=var)

#     if not os.path.isfile(data_file): # check if file already exists
#         print("Downloading %s from S3..." % s3_data_key)
#         client.download_file(era5_bucket, s3_data_key, data_file)
#         print("time to download file (seconds): ", time.time()-start2)

#     ds = xr.open_dataset(data_file)
#     #ds.info

#     # TAKE ERA5 DATA AND EXTRACT DATA FROM LAT/LON COMBO
    
#     ds_locs = xr.Dataset()
#     # interate through the locations and create a dataset
#     # containing the weather values for each location
#     print("Start Extraction: ", var)
#     start3=time.time()
#     for l in locs:
#         name = l['name']+'_'+var
#         lon = l['lon']
#         lat = l['lat']
#         var_name = name

#         ds2 = ds.sel(lon=lon, lat=lat, method='nearest')

#         lon_attr = '%s_lon' % name
#         lat_attr = '%s_lat' % name

#         ds2.attrs[lon_attr] = ds2.lon.values.tolist()
#         ds2.attrs[lat_attr] = ds2.lat.values.tolist()
#         ds2 = ds2.rename({var : var_name}).drop(('lat', 'lon'))

#         ds_locs = xr.merge([ds_locs, ds2])

#     # CONVERT TO DATAFRAME AND WRITE TO PARQUET
#     df=ds_locs.to_dataframe()
#     df.to_parquet('data/{}_{}.parquet'.format(date.strftime('%Y_%m'),var))
#     print("time to extract data (seconds): ", time.time()-start2)
#     print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# print("time to download all variables (secs): ", time.time()-start)
