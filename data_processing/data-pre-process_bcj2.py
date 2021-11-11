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
# Add:
# * distance calculation
# * reading count
# * qa_weightage
#
# Breakdown:
# * time units 
# * resolution 

# +
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import altair as alt


import argparse
import glob
import time
import sklearn
import numpy as np
import pandas as pd
import geojson
import json
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
from descartes import PolygonPatch

alt.data_transformers.disable_max_rows()
# -

# ### Operations to apply
#
# * https://heartbeat.comet.ml/working-with-geospatial-data-in-machine-learning-ad4097c7228d
#
#
# * `dist_away` - from rounded points to the true point
# * `reading_count` - When aggregating over a given spot, number of readings we have for that spot
# * instead of every day, change data granularity to every week, or every 10 days etc. this would make missing data problem "dissappear"
# * For row (if day then also include week, if week then also include month granularity)
# * potentially instead of lat/lon we can do x,y,z... (may be unnecessary since we are just focussed on CA)



# ### Toggles
# * resolution - `[raw, 0.1, 0.2, 0.5, 1.0]`
# * time - grouping any day increment `[ '1D', '3D', '5D', '7D', 10D']`
#
#
#
#
#
# #### Different Day Groupings
# * https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
# * https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

# ### Load Data

# +
start = time.time()

location_name = 'testsite3'

# s3_file_path = 's3://methane-capstone/data/combined-raw-data/combined-raw.parquet.gzip'
s3_file_path = f's3://methane-capstone/data/{location_name}/combined-raw-data/combined-raw-bcj.parquet'

df = pd.read_parquet(s3_file_path)
df['time_utc'] = pd.to_datetime(df['time_utc'])
df['year_month'] = df.year_month.astype(str)
print(df.shape)
print(df.dtypes)
end = time.time()
print("Load time", end-start)
# -
# np.nanmean([np.nan,np.nan,np.nan,np.nan,1,2,3,4])  #testing out means
# np.mean([1,2,3,4])  #testing out means
print(df.columns)
df.head()


#Helper function to calculate distance between raw lat/lon and rounded lat/lon
def haversine_distance(row, rounded_pair, unit = 'km'):

    lat_s, lon_s = row['lat'], row['lon'] #Source
    lat_d, lon_d = row[rounded_pair[0]], row[rounded_pair[1]] #Rounded Point - row['Destination Lat'], row['Destination Long']
    radius = 6371 if unit == 'km' else 3956 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.

    dlat = np.radians(lat_d - lat_s)
    dlon = np.radians(lon_d - lon_s)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat_s)) * np.cos(np.radians(lat_d)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c
    return distance


# ### Process data by resolution and time!
#
# * note right now we are not doing anything with `qa_val`

# +
def get_time_resolution_groups(df, freq, resolution):
    
    bins = []
    groups = []
    methane_columns = ['time_utc', 'year_month', 'lat', 'lon', 'rn_lat_1', 'rn_lon_1',
                   'rn_lat_2', 'rn_lon_2', 'rn_lat_5', 'rn_lon_5', 'rn_lat', 'rn_lon',
                   'qa_val', 'methane_mixing_ratio', 'methane_mixing_ratio_precision',
                   'methane_mixing_ratio_bias_corrected']

    weather_columns = ['air_pressure_at_mean_sea_level', 'air_temperature_at_2_metres',
               'air_temperature_at_2_metres_1hour_Maximum',
               'air_temperature_at_2_metres_1hour_Minimum',
               'dew_point_temperature_at_2_metres', 'eastward_wind_at_100_metres',
               'eastward_wind_at_10_metres',
               'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation',
               'lwe_thickness_of_surface_snow_amount', 'northward_wind_at_100_metres',
               'northward_wind_at_10_metres',
               'precipitation_amount_1hour_Accumulation', 'snow_density',
               'surface_air_pressure']
       
    well_columns = ['well_count_rn_1', 'well_count_rn_2', 'well_count_rn_5', 'well_count_rn']
    
    if freq == "1D":
        if resolution == 0.1:
            resolution_gb_cols = ['rn_lat_1','rn_lon_1']
#            well_col = 'well_count_rn_1'
  #          dist_away_col = 'dist_away_1'
        elif resolution == 0.2:
            resolution_gb_cols = ['rn_lat_2','rn_lon_2']
#            well_col = 'well_count_rn_2'
  #          dist_away_col = 'dist_away_2'
        elif resolution == 0.5:
            resolution_gb_cols = ['rn_lat_5','rn_lon_5']
#            well_col = 'well_count_rn_5'
   #         dist_away_col = 'dist_away_5'
        elif resolution == 1.0:
            resolution_gb_cols = ['rn_lat','rn_lon']
#            well_col = 'well_count_rn'
   #         dist_away_col = 'dist_away'
  

        
        #Distance away column
        #df[dist_away_col] = df.apply(lambda x: haversine_distance(x, resolution_gb_cols), axis=1)
        
        df_reduced = df.groupby([df.time_utc.dt.date] + resolution_gb_cols).agg({'methane_mixing_ratio': ["count","mean"],
                                                            'methane_mixing_ratio_precision':"mean",
                                                            'methane_mixing_ratio_bias_corrected': "mean",
                                                           # dist_away_col: ["mean"],
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
                                                           # well_col: ['mean'],               
                                                            'qa_val': [('mode',lambda x:x.value_counts().index[0]), "mean"], #Numerical = Weight, Categorical = Mode
                                                           })
        
       
        #Flatten MultiIndex
        df_reduced.columns = ['_'.join(col) for col in df_reduced.columns.values]
        df_reduced = df_reduced.reset_index()
        df_reduced = df_reduced.rename(columns={"methane_mixing_ratio_count": "reading_count"})
        
        groups = [df_reduced]
        bins = df_reduced['time_utc'].tolist()

        print("Num groups", len(df_reduced))
        
    else:
    
        for bin_start_dt, df_grp in df.set_index('time_utc').groupby(pd.Grouper(freq=freq, origin="start")):
   
            #print(bin_start_dt)     
    
            if resolution == 0.1:
                resolution_gb_cols = ['rn_lat_1','rn_lon_1']
  #              well_col = 'well_count_rn_1'
   #             dist_away_col = 'dist_away_1'
            elif resolution == 0.2:
                resolution_gb_cols = ['rn_lat_2','rn_lon_2']
   #             well_col = 'well_count_rn_2'
    #            dist_away_col = 'dist_away_2'
            elif resolution == 0.5:
                resolution_gb_cols = ['rn_lat_5','rn_lon_5']
    #            well_col = 'well_count_rn_5'
    #            dist_away_col = 'dist_away_5'
            elif resolution == 1.0:
                resolution_gb_cols = ['rn_lat','rn_lon']
     #           well_col = 'well_count_rn'
     #           dist_away_col = 'dist_away'               
                
            #Distance away column
            #df_grp[dist_away_col] = df_grp.apply(lambda x: haversine_distance(x, resolution_gb_cols), axis=1)
        

#         To also get the number of readings we have each day
#         df_grp_reading_count = df_grp.groupby(resolution_gb_cols).agg({
#                                                              'methane_mixing_ratio': 'count',
#                                                             }).reset_index().rename(columns={"methane_mixing_ratio": "reading_count"})
#         df_grp = df_grp.merge(df_grp_reading_count, how='inner', left_on=resolution_gb_cols, right_on = resolution_gb_cols)
#         qa_vals_count = df_grp['qa_val'].value_counts().to_dict()

            #option 1 - only keep bad data if no other data exists, 
            #option 2 - interpolate bad data
        
                    
            df_reduced = df_grp.groupby(resolution_gb_cols).agg({'methane_mixing_ratio': ["count","mean"],
                                                                'methane_mixing_ratio_precision':"mean",
                                                                'methane_mixing_ratio_bias_corrected': "mean",
                                                               # dist_away_col: ["mean"],
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
                                                               # well_col: ['mean'],  
                                                                'qa_val': [('mode',lambda x:x.value_counts().index[0]), "mean"], #Numerical = Weight, Categorical = Mode
                                                               })

            #Flatten MultiIndex
            df_reduced.columns = ['_'.join(col) for col in df_reduced.columns.values]
            df_reduced = df_reduced.reset_index()
            df_reduced = df_reduced.rename(columns={"methane_mixing_ratio_count": "reading_count"})

            
            #Add in date-time
            df_reduced.insert(0, "time_utc", value=[bin_start_dt]*len(df_reduced))
            df_reduced['time_utc'] = pd.to_datetime(df_reduced['time_utc'])

            bins.append(bin_start_dt)
            groups.append(df_reduced)

#         print()
        print("Num groups", len(groups))
    
    return groups, bins


def get_processed_df(df, freq, resolution):

    groups, bins = get_time_resolution_groups(df, freq, resolution)
    
    columns = groups[0].columns.tolist()
    df_final = pd.DataFrame(columns=columns)
    
    for ind, group in enumerate(groups, 1):
#         print(f'#{ind}, shape: {group.shape}')
        df_final = pd.concat([df_final, group])
        
    df_final = df_final.sort_values('time_utc')

    print()
    print("final shape", df_final.shape)
    return df_final



bucket = 'methane-capstone'
subfolder = f'data/{location_name}/data-variants2'
s3_path = bucket+'/'+subfolder


def write_to_s3(dataframe, freq, reso):
    if reso == 0.1:
        reso_str = 'rn_1'
    elif reso == 0.2:
        reso_str = 'rn_2'
    elif reso == 0.5:
        reso_str = 'rn_5'
    elif reso == 1.0:
        reso_str = 'rn'
    else:
        raise Error

    file_name=f'data_{freq}_{reso_str}.parquet.gzip'
    file_path = 's3://{}/{}'.format(s3_path, file_name)
    print(file_path)
    dataframe.to_parquet(file_path, compression='gzip')
    


# -
# ### Run Data Processing

5+5

# +
RESOLUTIONS = [0.1, 0.2, 0.5, 1.0]
FREQUENCIES = [ '1D', '3D', '5D', '7D', '10D']

for FREQUENCY in FREQUENCIES:
    for RESOLUTION in RESOLUTIONS:
        print("RESOLUTION", RESOLUTION, "FREQUENCY", FREQUENCY)
        
        start = time.time()
        df_cur = get_processed_df(df, FREQUENCY, RESOLUTION)
        write_to_s3(df_cur, FREQUENCY, RESOLUTION)

        print("total time", time.time() - start)
# -


5+5


# ### Save to S3

# write_to_s3(df_final, FREQUENCY, RESOLUTION)

#

# ### Check Number of Unique Geo spots

print(df[['lat','lon','rn_lat_1', 'rn_lon_1', 'rn_lat_2' ,'rn_lon_2' ,'rn_lat_5', 'rn_lon_5', 'rn_lat', 'rn_lon']].nunique())
a1, a2, b1, b2, c1, c2, d1, d2, e1, e2 = df[['lat','lon','rn_lat_1', 'rn_lon_1', 'rn_lat_2' ,'rn_lon_2' ,'rn_lat_5', 'rn_lon_5', 'rn_lat', 'rn_lon']].nunique()
print()
print("Number of locations:")
print("raw", '{:,}'.format(a1*a2))
print("raw - 0.1", '{:,}'.format(b1*b2))
print("raw - 0.2", '{:,}'.format(c1*c2))
print("raw - 0.5", '{:,}'.format(d1*d2))
print("raw - 1.0", '{:,}'.format(e1*e2))



# ### Check Data Distribution

df_sample = df.sample(frac=0.005)
print(df_sample.shape)
df_sample.head()

boom = df_sample['methane_mixing_ratio_bias_corrected'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Methane Mixing Ratio - CA')
plt.xlabel('PPB')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)

boom


# -

boom = df_sample['dist_away_5_mean'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Methane Mixing Ratio - CA')
plt.xlabel('dist_away_5_mean')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)

# +
print(df_other['reading_count'].value_counts())

boom = df_other['reading_count'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Methane Mixing Ratio - CA')
plt.xlabel('reading_count')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)
# -

boom = df_other['qa_val_mode'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Methane Mixing Ratio - CA')
plt.xlabel('QA_MODE')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)



boom = df_other[df_other['qa_val_mean'] < 0.3999]['qa_val_mean'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Methane Mixing Ratio - CA')
plt.xlabel('QA_WEIGHTAGED')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.75)



