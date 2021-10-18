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
s3_file_path = 's3://methane-capstone/data/combined-raw-data/combined-raw.parquet.gzip'

df = pd.read_parquet(s3_file_path)
df['time_utc'] = pd.to_datetime(df['time_utc'])
df['year_month'] = df.year_month.astype(str)
print(df.shape)
print(df.dtypes)
end = time.time()
print("Load time", end-start)

# -



# +
#Read new file
file_name = '/root/methane/data_processing/resources/vista-ca-combined'
gdf = gpd.read_file("{}.shp".format(file_name))
gdf['area'] = gdf['geometry'].to_crs({'init': 'epsg:3395'})\
               .map(lambda p: p.area / 10**6)


print(gdf.shape)
gdf.head()
# -

print(gdf[gdf['vistastype'] != 'Oil and Gas Well'].shape)
gdf.groupby('vistastype').agg({'area': ['mean','count']})

# ### Plot CA

# +
geo_json_path = "../data_processing/resources/california.geojson"
with open(geo_json_path) as json_file:
    geojson_data = geojson.load(json_file)
ca_poly = geojson_data['geometry']

ca_gdf = gpd.read_file(geo_json_path)
choro_json = json.loads(ca_gdf.to_json())
choro_data = alt.Data(values=choro_json['features'])

# Create Base CA Map
ca_base = alt.Chart(choro_data, title = 'California ').mark_geoshape(
    color='lightgrey',
    opacity=0.3,
    stroke='black',
    strokeWidth=1
).encode().properties(
    width=500,
    height=500
)

# -

gdf.head()

gdf.columns

# +
gdf_oil_wells = gdf[gdf['vistastype'] == 'Oil and Gas Well']
gdf_trim = gdf[gdf['vistastype'] != 'Oil and Gas Well']

print(gdf_oil_wells.shape)
print(gdf_trim.shape)


# +
points = alt.Chart(gdf_oil_wells[:30000]).mark_circle(size=10).encode(
    longitude=f'longitude:Q',
    latitude=f'latitude:Q',
    tooltip= 'vistastype',
    color='vistastype:N'
)

ca_base + points

# +
#Plot all the readings
gdf_trim_plot= gdf_trim[['latitude', 'longitude', 'vistastype', 'area']]

points = alt.Chart(gdf_trim_plot).mark_circle(size=10).encode(
    longitude=f'longitude:Q',
    latitude=f'latitude:Q',
    tooltip= 'vistastype',
    color='vistastype:N'
)

ca_base + points
# -



# ### Trim data to be not over the sites

df.shape

gdf.head()

# +
# List of all facilities polygons

vista_type_list = gdf_trim['vistastype'].tolist()
poly_list = gdf_trim['geometry'].tolist()

print(len(vista_type_list))
print(len(poly_list))

# +
# def isInAFacility(row):
#     '''Given lat, lon. Return Boolean if in California'''
#     point = Point(row['lon'], row['lat'])
    
#     print(row.index)
#     for poly in poly_list:
#         if poly.contains(point):
#             return True
#     return False


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
type_and_inFacility = []

start = time.time()
for point in tqdm(processed_points):
    
    found=False
    
    for i, poly in enumerate(poly_list, 0):
        if poly.contains(point):
            type_and_inFacility.append((str(vista_type_list[i]), True))
            found = True
            
            #If point has been found, no need to look at other polys
            break

    if not found:   
        type_and_inFacility.append((None, False))

end = time.time()

# +

print(end-start)
# -

print(len(type_and_inFacility))

# +
# df[['lat','lon']][:10].apply(isInAFacility, axis=1)
# -

df['point_type'] = [tup[0] for tup in tqdm(type_and_inFacility)]


df['inFacility'] = [tup[1] for tup in tqdm(type_and_inFacility)]

# +
# start = time.time()
# df['inFacility'] = df[['lat','lon']].apply(isInAFacility, axis=1)
# end = time.time()
# print(end-start)

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

write_to_s3(df, 'combined-raw-facility-details')

df.groupby('point_type').size()

df.groupby('inFacility').size()

3381/(1215953+3381)

# ### Merge Oil and Gas Well Points
#
# * All these places we do not have polygons, only points

gdf_oil_wells.head()

lat_sorted.iloc[-100:,:]

gdf[gdf['vistastype'] == "Natural Gas Station"]

# +
sample_df = pd.DataFrame(columns=['latitude','longitude','vistastype','geometry','area'])
lat_sorted = gdf_oil_wells.sort_values('latitude')

interval = 15000
midpoint = int(lat_sorted.shape[0]/2)

print(midpoint)
print(lat_sorted.shape)
print()
sample_df = pd.concat([lat_sorted.iloc[:interval,:], lat_sorted.iloc[midpoint:midpoint+interval,:], lat_sorted.iloc[-interval:,:]])

# +
oil_points = alt.Chart(sample_df).mark_circle(size=10).encode(
    longitude=f'longitude:Q',
    latitude=f'latitude:Q',
    tooltip= 'vistastype',
    color='vistastype:N'
)

ca_base + oil_points



# +
# from shapely.ops import cascaded_union
# polygons = [poly1[0], poly1[1], poly2[0], poly2[1]]
# boundary = gpd.GeoSeries(cascaded_union(polygons))
# boundary.plot(color = 'red')
# plt.show()
# -

vals = [type(x) for x in gdf_oil_wells.geometry]
Counter(vals)


