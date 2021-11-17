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

# ### Understand Climate Zones
# * https://cecgis-caenergy.opendata.arcgis.com/datasets/CAEnergy::california-building-climate-zones/explore?location=37.062390%2C-120.193659%2C5.99
# * https://www.pge.com/includes/docs/pdfs/about/edusafety/training/pec/toolbox/arch/climate/california_climate_zones_01-16.pdf

#



# +
import pandas as pd
import numpy as np
import geopandas as gpd
import geojson
import json
import altair as alt 
import time
from datetime import datetime
from tqdm import tqdm
from collections import Counter

#GeoPandas
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

alt.data_transformers.disable_max_rows()
# -

# ### All Data

# +
start = time.time()
s3_file_path = 's3://methane-capstone/data/combined-raw-data/combined-raw-facility-oil-weather.parquet.gzip'


df = pd.read_parquet(s3_file_path)
df['time_utc'] = pd.to_datetime(df['time_utc'])
df['year_month'] = df.year_month.astype(str)
print(df.shape)
print(df.dtypes)
end = time.time()
print("Load time", end-start)

# -



#Read new file
file_name = '/root/methane/data_processing/resources/ca_building_climate_zones.geojson'
cl_gdf = gpd.read_file(file_name)
cl_gdf

# +
city_rep = {    
    '1': 'Arcata',
    '2': 'Santa Rosa',
    '3': 'Oakland',
    '4': 'San Jose-Reid',
    '5': 'Santa Maria',
    '6': 'Torrance',
    '7': 'San Diego-Lindbergh',
    '8': 'Fullerton',
    '9': 'Burbank-Glendale',
    '10':'Riverside',
    '11': 'Red Bluff',
    '12':'Sacramento',
    '13':'Fresno',
    '14':'Palmdale',
    '15':'Palm Spring-Intl',
    '16':'Blue Canyon'    
}

cl_gdf.insert(1, 'rep_city', [city_rep[x] for x in cl_gdf['BZone']])
cl_gdf['center_lat'] = cl_gdf.geometry.centroid.y
cl_gdf['center_lon'] = cl_gdf.geometry.centroid.x
cl_gdf.head()
# -

# ### Plot CA

# +
geo_json_path = "../data_processing/resources/california.geojson"
with open(geo_json_path) as json_file:
    geojson_data = geojson.load(json_file)
ca_poly = geojson_data['geometry']

ca_gdf = gpd.read_file(geo_json_path)
ca_choro_json = json.loads(ca_gdf.to_json())
ca_choro_data = alt.Data(values=ca_choro_json['features'])

# Create Base CA Map
ca_base = alt.Chart(ca_choro_data).mark_geoshape(
    color='lightgrey',
    opacity=0.3,
    stroke='black',
    strokeWidth=1
).encode().properties(
    width=500,
    height=500
)
# -



# ### Plot Regions



# +
def open_geojson(geo_json_file_loc):
    with open(geo_json_file_loc) as json_data:
        d = json.load(json_data)
    return d

def get_gpd_df(geo_json_file_loc):
    chicago_json = open_geojson(geo_json_file_loc)
    gdf = gpd.GeoDataFrame.from_features((chicago_json))
    return gdf

def gen_map(geodata, chart_title, color_column, legend_title, tooltip, color_scheme='bluegreen', label=''):
    '''
    Generates Chicago Area map with car theft count choropleth
    '''
    # Add Base Layer
    base = alt.Chart(geodata, title = chart_title).mark_geoshape(
        color='lightgrey',
        opacity=0.3,
        stroke='black',
        strokeWidth=1
    ).encode().properties(
        width=800,
        height=800
    )
    
    # Add Choropleth Layer
    choro = alt.Chart(geodata).mark_geoshape(
        stroke='black'
    ).encode(
        alt.Color(color_column,
                  scale=alt.Scale(scheme=color_scheme),
                  title = legend_title),
        tooltip=tooltip
    )

    if label:
        # Add Labels Layer
        labels = alt.Chart(cl_gdf).mark_text(baseline='top'
        ).properties(
          width=400,
          height=400
        ).encode(
             longitude='center_lon:Q',
             latitude='center_lat:Q',
             text='BZone:N',
             size=alt.value(10),
             opacity=alt.value(1)
         )
        return base + choro + labels
    else:
        return base + choro
# -



# +
#https://www.districtdatalabs.com/altair-choropleth-viz

cl_choro_json = json.loads(cl_gdf.to_json())
cl_choro_data = alt.Data(values=cl_choro_json['features'])

# Create Base CA Map
climate_regions = alt.Chart(cl_choro_data, title = 'CA Climate Regions').mark_geoshape(
    opacity=0.3,
    stroke='black',
    strokeWidth=1
).encode(
    color = 'properties.BZone:N'
).properties(
    width=500,
    height=500
)


labels = alt.Chart(cl_gdf).mark_text().properties(
    width=400,
    height=400
 ).encode(
     longitude='center_lon:Q',
     latitude='center_lat:Q',
     text='BZone:N',
     size=alt.value(10),
     opacity=alt.value(1)
 )

# -

climate_regions + labels

# +
# df[df['time_utc'] < datetime(2018, 11, 29)]
# -



# ### Trim data to be not over the sites

# +
# List of all facilities polygons
zone_id_list = cl_gdf['BZone'].tolist()
region_poly_list = cl_gdf['geometry'].tolist()


print(len(region_poly_list))
print(len(zone_id_list))

# +
lats = df['lat'].tolist()
lons = df['lon'].tolist()

print(len(lats))
print(len(lons))


# +
def process_points(lon, lat):
    return Point(lon, lat)

processed_points = [process_points(lons[i], lats[i]) for i in tqdm(range(len(lats)))]


# +
point_zones = []

start = time.time()
for point in tqdm(processed_points):
    
    found=False
    
    for i, poly in enumerate(region_poly_list, 0):
        if poly.contains(point):
            point_zones.append(zone_id_list[i])
            found = True
            
            #If point has been found, no need to look at other polys
            break

    if not found:   
        point_zones.append(None)

end = time.time()
# -

df['BZone'] = point_zones

# +
bucket = 'methane-capstone'
subfolder = 'data/combined-raw-data'
s3_path = bucket+'/'+subfolder


def write_to_s3(dataframe, file_name):

    file_name=f'{file_name}.parquet.gzip'
    file_path = 's3://{}/{}'.format(s3_path, file_name)
    print(file_path)
    dataframe.to_parquet(file_path, compression='gzip')


# -

write_to_s3(df, 'combined-raw-facility-oil-weather')





# ## Region EDA 





df_zone_split = df.groupby('BZone').size().reset_index().rename({0:"count"}, axis=1)
df_zone_split['percent'] = df_zone_split['count']*100/ df_zone_split['count'].sum()
df_zone_split = df_zone_split.sort_values(by='percent', ascending=False)
df_zone_split

# + active=""
# climate_regions + labels
# -




# ### Missing Region Points:

missing_zones.set_index("time_utc").groupby(pd.Grouper(freq="Y")).size()

# +
miss_ym_group = missing_zones.groupby(df.time_utc.dt.date).size().reset_index().rename({0:"count"}, axis=1)
miss_ym_group['time_utc'] = pd.to_datetime(miss_ym_group['time_utc'])
daily = alt.Chart(miss_ym_group, title="Unknown Regions - Daily Count").mark_line(
    point={
          "filled": False,
          "fill": "white"
        }
    ).encode(
        x='yearmonthdate(time_utc):O',
        y=alt.Y('mean(count)', title='Daily Mean'),
        tooltip=['yearmonthdate(time_utc):O', 'mean(count)']
    )

monthly = alt.Chart(miss_ym_group, title="Unknown Regions - Monthly Count").mark_line(
    point={
          "filled": False,
          "fill": "white"
        }
    ).encode(
        x='yearmonth(time_utc):O',
        y=alt.Y('sum(count)', title='Monthly Count'),
        tooltip=['yearmonth(time_utc):O', 'sum(count)']
    )

daily & monthly
# -



# ### Weather Graph Demo

# +
# weaeth_path = 's3://methane-capstone/data/combined-raw-data/combined-raw-facility-oil-weather.parquet.gzip'
# wdf = pd.read_parquet(weaeth_path)
# wdf['time_utc'] = pd.to_datetime(df['time_utc'])
# wdf['year_month'] = df.year_month.astype(str)
# wdf_trim = wdf[['time_utc','year_month','lat','lon','eastward_wind_at_10_metres']]

# wdf_2021 = wdf_trim[wdf_trim.time_utc.dt.year > 2020]
# print(wdf_2021.shape)

# wdf_2021_samp = wdf_2021.sample(n=30000)

# wind_points = alt.Chart(wdf_2021_samp, title='Wind Patterns - CA 2021').mark_circle(size=30).encode(
#     longitude=f'lon:Q',
#     latitude=f'lat:Q',
#     tooltip= 'year_month:O',
#     color=alt.Color('eastward_wind_at_10_metres:Q', scale=alt.Scale(scheme = 'redblue'))
# )

# ca_base + wind_points
# -



# +
missing_zones = df[df['BZone'].isnull()]
print(missing_zones.shape)

miss_points = alt.Chart(missing_zones).mark_circle(size=30).encode(
    longitude=f'lon:Q',
    latitude=f'lat:Q',
    tooltip= 'year_month:O',
    color='year(time_utc):N'
)

ca_base + miss_points
# -



#

# ### Reading Counts

dt_zone_count_day = df.groupby([df.time_utc.dt.date, 'BZone']).size().reset_index().rename({0:"count"}, axis=1)
dt_zone_count_day['time_utc'] = pd.to_datetime(dt_zone_count_day['time_utc'])
dt_zone_count_day

dt_zone_count_month = df.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']).size().reset_index().rename({0:"count"}, axis=1)
dt_zone_count_month['time_utc'] = pd.to_datetime(dt_zone_count_month['time_utc'])
dt_zone_count_month

#Average number of readings per day, for each zone
# missing_zones.set_index("time_utc").groupby(pd.Grouper(freq="Y")).size()
zone_count_day_avg = dt_zone_count_day.groupby('BZone').agg({"count": "mean"}).sort_values('count').reset_index()
zone_count_day_avg

# +
selector = alt.selection_multi(empty='all', fields=['BZone'])

base = alt.Chart(cl_gdf[['rep_city', 'BZone', 'SHAPE_Area', 'center_lat', 'center_lon']]).properties(
    width=250,
    height=250
).add_selection(selector)

points = base.mark_point(filled=True, size=200).encode(
    x=alt.X('center_lon', scale=alt.Scale(zero=False)),
    y = alt.Y('center_lat', scale=alt.Scale(zero=False)),
    tooltip=['BZone','rep_city', 'center_lat', 'center_lon'],
    color=alt.condition(selector, 'BZone:N', alt.value('lightgray'), legend=None),
)

area_bar = base.mark_bar().encode(
    x = alt.X('BZone:N'),
    y = alt.Y('SHAPE_Area'),
    tooltip=['BZone','rep_city', 'SHAPE_Area'],
    color=alt.condition(selector, 'BZone:N', alt.value('lightgray'), legend=None),
)

month_avg_bar = alt.Chart(dt_zone_count_month).mark_bar().encode(
    x = alt.X('BZone:N'),
    y = alt.Y('count:Q', title='Monthly Average'),
    tooltip=['BZone','count:Q'],
    color=alt.condition(selector, 'BZone:N', alt.value('lightgray'), legend=None),
).add_selection(selector)


day_avg_bar = alt.Chart(dt_zone_count_day).mark_bar().encode(
    x = alt.X('BZone:N'),
    y = alt.Y('count:Q', title='Day Average'),
    tooltip=['BZone','count:Q'],
    color=alt.condition(selector, 'BZone:N', alt.value('lightgray'), legend=None),
).add_selection(selector)

#### #### #### #### #### 

region_by_month = alt.Chart(dt_zone_count_month, title="Monthly Reading Count").mark_line(
    point={
          "filled": False,
          "fill": "white"
        }
    ).encode(
        x='yearmonth(time_utc):O',
        y=alt.Y('count', title='Monthly Count'),
        tooltip=['yearmonth(time_utc):O', 'count', 'BZone'],
        color=alt.Color('BZone:N') #, legend=None)
    ).transform_filter(
        selector
    ).add_selection(selector)



#https://www.districtdatalabs.com/altair-choropleth-viz

# cl_choro_json = json.loads(cl_gdf.to_json())
# cl_choro_data = alt.Data(values=cl_choro_json['features'])


(points & area_bar) | (region_by_month & (month_avg_bar | day_avg_bar))
# -

#

# ### Understanding Methane

dt_zone_meth_day = df.groupby([df.time_utc.dt.date, 'BZone']).agg({'methane_mixing_ratio_bias_corrected': 'mean'}).reset_index()
dt_zone_meth_day['time_utc'] = pd.to_datetime(dt_zone_meth_day['time_utc'])
dt_zone_meth_day

dt_zone_meth_month = df.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']).agg({'methane_mixing_ratio_bias_corrected': 'mean'}).reset_index()
dt_zone_meth_month['time_utc'] = pd.to_datetime(dt_zone_meth_day['time_utc'])
dt_zone_meth_month

# +
selector2 = alt.selection_multi(empty='all', fields=['BZone'])

base = alt.Chart(cl_gdf[['rep_city', 'BZone', 'SHAPE_Area', 'center_lat', 'center_lon']]).properties(
    width=250,
    height=250
).add_selection(selector2)

avg_meth_bar = alt.Chart(dt_zone_meth_month, title='Monthly Zone Average').mark_bar().encode(
    x = alt.X('BZone:N'),
    y = alt.Y('mean(methane_mixing_ratio_bias_corrected):Q',
              scale=alt.Scale(zero=False)),
    tooltip=['BZone', 'mean(methane_mixing_ratio_bias_corrected):Q'],
    color=alt.condition(selector2, 'BZone:N', alt.value('lightgray'), legend=None)
).add_selection(selector2)


points = base.mark_point(filled=True, size=200).encode(
    x=alt.X('center_lon', scale=alt.Scale(zero=False)),
    y = alt.Y('center_lat', scale=alt.Scale(zero=False)),
    tooltip=['BZone','rep_city', 'center_lat', 'center_lon'],
    color=alt.condition(selector2, 'BZone:N', alt.value('lightgray'), legend=None),
)

region_meth_month = alt.Chart(dt_zone_meth_day, title="Monthly Methane Avg").mark_line(
    point={
          "filled": False,
          "fill": "white"
        }
    ).encode(
        x='yearmonth(time_utc):O',
        y=alt.Y('mean(methane_mixing_ratio_bias_corrected)', title='Monthly Methane', scale=alt.Scale(zero=False)),
        tooltip=['yearmonth(time_utc):O', 'mean(methane_mixing_ratio_bias_corrected)', 'BZone'],
        color=alt.Color('BZone:N')
    ).transform_filter(
        selector2
    ).add_selection(selector2)

(points & avg_meth_bar) | region_meth_month

# -





df.shape

df


