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

# ### Explore Page Structure:
#
# #### Description Blurb
#
# #### Zone Meta - Section
#     * Purpose of this section is to provide high level meta data/ information about the zones
#     * NOTHING to do with comparing features across zones
#     
#     #### Charts/Items
#         * Table with all Zone Details:
#             * id, name, acerage, area, centerLat, centerLon
#         * Mapbox of CA with Zone split
#         * Bar Chart - Number of readings by region (Percent or Count)
#
# #### Feature Comparison
#     * Feature Selection
#     * Lat/Lon Scatter Selection
#     * Line Plot against Chosen Zones
#     * Time Aggregation
#     
#     
# ### Lat/Lon Exploration 
#     * Share the points plotted over time 
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


df_all = pd.read_parquet(s3_file_path)
df_all['time_utc'] = pd.to_datetime(df_all['time_utc'])
df_all['year_month'] = df_all.year_month.astype(str)
df_all['BZone'] = pd.to_numeric(df_all['BZone'])
print(df_all.shape)
print(df_all.dtypes)
end = time.time()
print("Load time", end-start)


# -
start = time.time()
s3_file_path = 's3://methane-capstone/data/combined-raw-data/data-zone-combined.parquet.gzip'
df_zone = pd.read_parquet(s3_file_path)
df_zone['time_utc'] = pd.to_datetime(df_zone['time_utc'])
df_zone['BZone'] = pd.to_numeric(df_zone['BZone'])
print(df_zone.shape)
print(df_zone.dtypes)
end = time.time()
print("Load time", end-start)




# +
#Read new file
file_name = '/root/methane/data_processing/resources/ca_building_climate_zones.geojson'
cl_gdf = gpd.read_file(file_name)
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

cl_gdf.insert(2, 'rep_city', [city_rep[x] for x in cl_gdf['BZone']])
cl_gdf['BZone'] = pd.to_numeric(cl_gdf['BZone'])
cl_gdf['center_lat'] = cl_gdf.geometry.centroid.y
cl_gdf['center_lon'] = cl_gdf.geometry.centroid.x
cl_gdf = cl_gdf.sort_values('BZone', ascending=True)

cl_gdf.head()



# -

# ### Format Table

# +
cl_gdf_cp = cl_gdf.copy()


cl_gdf_cp = cl_gdf_cp.drop(['FID', 'geometry', 'SHAPE_Length'], axis=1)
cl_gdf_cp = cl_gdf_cp.rename({"BZone": "id", 
                  'rep_city': 'name',
                  'BAcerage': "acerage",
                  'SHAPE_Area': "area",
                  'center_lat': "centerLat",
                  'center_lon': "centerLon",
                 }, axis=1).sort_values("id")

cl_gdf_cp = pd.DataFrame(cl_gdf_cp)

# +
# To JSON
# cl_gdf_cp.to_json('zone-meta.json', orient='records')
# -



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

climate_regions + labels

# +
# df[df['time_utc'] < datetime(2018, 11, 29)]
# -





# ## Region EDA 

df_zone_split = df_all.groupby('BZone').size().reset_index().rename({0:"count"}, axis=1)
df_zone_split['percent'] = df_zone_split['count']*100/ df_zone_split['count'].sum()
df_zone_split = df_zone_split.rename({"BZone": "id", 
                  'count': 'reading_count',
                 }, axis=1).sort_values(by='id', ascending=True)
df_zone_split

# +
zone_count_bar = alt.Chart(df_zone_split).mark_bar(tooltip=True).encode(x= alt.X('id:N'),
                                                                        y= alt.X('percent:Q'),
                                                                        tooltip=['id', 'reading_count', 'percent'])

# text = zone_count_bar.mark_text(
#     align='left',
#     baseline='middle',
#     dx=5  # Nudges text to right so it doesn't appear on top of the bar
# ).encode(
#     text='percent:Q'
# )

# (zone_count_bar + text).properties(height=900)

zone_count_bar
# -

zone_count_bar.save('zone_count_bar.json')




# ### Weather Graph Demo

feature_cols = ['methane_mixing_ratio_bias_corrected_mean',
                 'reading_count',
                 'air_pressure_at_mean_sea_level_mean',
                 'eastward_wind_at_100_metres_mean',
                 'northward_wind_at_100_metres_mean',
                 'air_temperature_at_2_metres_mean',
                 'surface_air_pressure_mean',
                 'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean',
                 'precipitation_amount_1hour_Accumulation_mean',
                 'dew_point_temperature_at_2_metres_mean'
               ]
id_cols = ['time_utc', 'BZone']
df_zone = df_zone[id_cols + feature_cols]
print(df_zone.shape)
df_zone.head()







# ### Vista CA

# +
#Read new file
file_name = '/root/methane/data_processing/resources/vista-ca-combined'
vista_gdf = gpd.read_file("{}.shp".format(file_name))
vista_gdf['area'] = vista_gdf['geometry'].to_crs({'init': 'epsg:3395'})\
               .map(lambda p: p.area / 10**6)

lats = np.array(vista_gdf['latitude'])
lons = np.array(vista_gdf['longitude'])
        
vista_gdf['rn_lat_1'] =  np.round(lats, 1)
vista_gdf['rn_lon_1'] =  np.round(lons, 1)

vista_gdf['rn_lat_2'] =  np.round(lats*5)/5
vista_gdf['rn_lon_2'] =  np.round(lons*5)/5

vista_gdf['rn_lat_5'] =  np.round(lats*2)/2
vista_gdf['rn_lon_5'] =  np.round(lons*2)/2

vista_gdf['rn_lat'] =  np.round(lats, 0)
vista_gdf['rn_lon'] =  np.round(lons, 0)

print(vista_gdf.shape)
vista_gdf.head()

# +
vista_df = pd.DataFrame(vista_gdf[['latitude', 'longitude','vistastype', 'area',
                                    'rn_lat_1', 'rn_lon_1',
                                    'rn_lat_2', 'rn_lon_2',	
                                    'rn_lat_5', 'rn_lon_5'
                                   ]])
vista_df.head()


# -

non_oil_well = vista_df[vista_df['vistastype'] != 'Oil and Gas Well']
non_oil_well.shape



vista_gdf.vistastype.value_counts()

df_all.point_type.unique()

df_all.point_type.value_counts()





# ### Feature Comparison Dashboard
#
# * 'methane_mixing_ratio_bias_corrected_mean', 'reading_count'
# * 'air_pressure_at_mean_sea_level_mean'
# * 'eastward_wind_at_100_metres_mean'
# * 'northward_wind_at_100_metres_mean'
# * 'air_temperature_at_2_metres_mean', 'surface_air_pressure_mean'
# * 'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean'
# * 'precipitation_amount_1hour_Accumulation_mean'
# * 'dew_point_temperature_at_2_metres_mean'

# +
feature_name_map = {

    'methane_mixing_ratio_bias_corrected_mean': "Methane (ppb)",
    'reading_count': "Reading Count",
    'air_pressure_at_mean_sea_level_mean': "Sea Level Air Pressure (Pa)",
    'eastward_wind_at_100_metres_mean': "Eastward Wind (m/s)",
    'northward_wind_at_100_metres_mean': "Northward Wind (m/s)",
    'air_temperature_at_2_metres_mean': "Air Temperature (K)",
    'surface_air_pressure_mean': "Surface Air Pressure (Pa)",
    'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean': "Solar Radiation (J/m2)",
    'precipitation_amount_1hour_Accumulation_mean': "Precipitation (m)",
    'dew_point_temperature_at_2_metres_mean': "Dew Point Temperature (K)",
    'center_lat': "Center Latitude",
    'center_lon': "Center Longitude"
    
    
}


# -

compare_feature = feature_cols[0]
compare_feature

dt_zone_count_day = df_all.groupby([df_all.time_utc.dt.date, 'BZone']).size().reset_index().rename({0:"count"}, axis=1)
dt_zone_count_day['time_utc'] = pd.to_datetime(dt_zone_count_day['time_utc'])
dt_zone_count_day

dt_zone_count_month = df_all.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']).size().reset_index().rename({0:"count"}, axis=1)
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


(points & area_bar) | (region_by_month & (month_avg_bar | day_avg_bar))
# -

#





# ### Feature Comparison v2


df_all.columns

compare_feature

df_zone.columns

# +
dt_zone_by_month = df_zone.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']) \
                                                .agg({'reading_count': 'sum', 'methane_mixing_ratio_bias_corrected_mean': 'mean'}) \
                                                .reset_index() \
                                                .rename({'reading_count':feature_name_map['reading_count'],
                                                         'methane_mixing_ratio_bias_corrected_mean':feature_name_map['methane_mixing_ratio_bias_corrected_mean'],
                                                        }, axis=1)
dt_zone_by_month['time_utc'] = pd.to_datetime(dt_zone_by_month['time_utc'])


# -

dt_zone_by_month



# +
### NEED TO ADDRESS ISSUE IF FEATURES ARE THE SAME


def get_feature_dashboard(time_feature, bar_feature):
    
    
    zone_selector = alt.selection_multi(empty='all', fields=['BZone'])
    time_brush = alt.selection_interval(encodings=['x'])
    
    time_agg = 'mean'
    bar_agg = 'mean'    
    
    isSame = time_feature == bar_feature
    
    if isSame:
        
        if time_feature == 'reading_count':
            time_agg = 'sum'
        
        dt_zone_by_month = df_zone.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']) \
                                                    .agg({time_feature: time_agg}) \
                                                    .reset_index() \
                                                    .rename({time_feature: feature_name_map[time_feature] + " " + time_agg.capitalize(),
                                                            }, axis=1)
                                                          
                                                          
        df_other = df_zone.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']) \
                                                    .agg({bar_feature: bar_agg}) \
                                                    .reset_index() \
                                                    .rename({time_feature: feature_name_map[bar_feature] + " " + bar_agg.capitalize(),
                                                            }, axis=1)

        dt_zone_by_month[feature_name_map[bar_feature] + " " + bar_agg.capitalize()] = df_other[feature_name_map[bar_feature] + " " + bar_agg.capitalize()]
                                                          
        
    
    else:

        if time_feature =='reading_count':
            time_agg = 'sum'

        dt_zone_by_month = df_zone.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']) \
                                                        .agg({time_feature: time_agg,
                                                              bar_feature: bar_agg}) \
                                                        .reset_index() \
                                                        .rename({time_feature: feature_name_map[time_feature],
                                                                 bar_feature: feature_name_map[bar_feature]
                                                                }, axis=1)
    
    
    dt_zone_by_month['time_utc'] = pd.to_datetime(dt_zone_by_month['time_utc'])

    time_suffix = " " + time_agg.capitalize() if isSame else ""
    bar_suffix = " " + bar_agg.capitalize() if isSame else ""
    
    
    #Scatter Plot for Zone Selection
    scatter_lat_lon = alt.Chart(cl_gdf[['rep_city', 'BZone', 'SHAPE_Area', 'center_lat', 'center_lon']], title="Zone Selection").mark_point(filled=True, size=200).encode(
                            x=alt.X('center_lon', title = feature_name_map['center_lon'], scale=alt.Scale(zero=False)),
                            y = alt.Y('center_lat', title= feature_name_map['center_lat'], scale=alt.Scale(zero=False)),
                            tooltip=['BZone','rep_city', 'center_lat', 'center_lon'],
                            color=alt.condition(zone_selector, 'BZone:N', alt.value('lightgray'), legend=None),
                        ).properties(
                            width = 250,
                            height = 300
                        ).add_selection(zone_selector)


    region_by_month = alt.Chart(dt_zone_by_month, title="Monthly " + feature_name_map[time_feature]).mark_line(
        point={
              "filled": False,
              "fill": "white"
            }
        ).encode(
            x='yearmonth(time_utc):O',
            y=alt.Y(f'{feature_name_map[time_feature] + time_suffix}:Q', title=f'{feature_name_map[time_feature]}', scale=alt.Scale(zero=False)),
            tooltip=['time_utc:O', f'{feature_name_map[time_feature] + time_suffix}:Q', 'BZone'],
            color=alt.condition(zone_selector | time_brush, 'BZone:N', alt.value('lightgray'), legend=None),
        ).transform_filter(
            zone_selector
        ).add_selection(zone_selector).add_selection(time_brush)



    month_avg_bar = alt.Chart(dt_zone_by_month, title=f'Monthly Average {feature_name_map[bar_feature]}').mark_bar().encode(
        x = alt.X('BZone:N'),
        y = alt.Y(f'mean({feature_name_map[bar_feature] + bar_suffix}):Q', title=f'{feature_name_map[bar_feature]} Mean', scale=alt.Scale(zero=False)),
        tooltip=['BZone', f'mean({feature_name_map[bar_feature ]+ bar_suffix}):Q'],
        color=alt.condition(zone_selector, 'BZone:N', alt.value('lightgray'), legend=None),
    ).transform_filter(
            time_brush
    ).add_selection(zone_selector)


    chart = (scatter_lat_lon | (month_avg_bar)) & (region_by_month)        

    return chart
        
        
        
# -

get_feature_dashboard('reading_count', feature_cols[0]) #units

chart = get_feature_dashboard('reading_count', feature_cols[0])
chart

chart.save('ft_dash.json')

dt_zone_count_month = df_zone.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'BZone']).agg({'reading_count': 'sum'}).reset_index().rename({0:"count"}, axis=1)
dt_zone_count_month['time_utc'] = pd.to_datetime(dt_zone_count_month['time_utc'])
dt_zone_count_month



# +
zone_selector = alt.selection_multi(empty='all', fields=['BZone'])
time_brush = alt.selection_interval(encodings=['x'])


lat_lon_points = alt.Chart(cl_gdf[['rep_city', 'BZone', 'SHAPE_Area', 'center_lat', 'center_lon']]).mark_point(filled=True, size=200).encode(
                        x=alt.X('center_lon', scale=alt.Scale(zero=False)),
                        y = alt.Y('center_lat', scale=alt.Scale(zero=False)),
                        tooltip=['BZone','rep_city', 'center_lat', 'center_lon'],
                        color=alt.condition(zone_selector, 'BZone:N', alt.value('lightgray'), legend=None),
                    ).properties(
                        width=250,
                        height=250
                    ).add_selection(zone_selector)


region_by_month = alt.Chart(df_zone, title="Monthly Reading Count").mark_line(
    point={
          "filled": False,
          "fill": "white"
        }
    ).encode(
        x='time_utc:O',
        y=alt.Y('reading_count:Q', title='Monthly Count'),
        tooltip=['time_utc:O', '(reading_count):Q', 'BZone'],
#         color=alt.Color('BZone:N') #, legend=None)
        color=alt.condition(zone_selector | time_brush, 'BZone:N', alt.value('lightgray'), legend=None),
    ).transform_filter(
        zone_selector
    ).add_selection(zone_selector).add_selection(time_brush)



month_avg_bar = alt.Chart(df_zone).mark_bar().encode(
    x = alt.X('BZone:N'),
    y = alt.Y('mean(reading_count):Q', title='Monthly Average'),
    tooltip=['BZone','mean(reading_count):Q'],
    color=alt.condition(zone_selector, 'BZone:N', alt.value('lightgray'), legend=None),
).transform_filter(
        time_brush
).add_selection(zone_selector)


day_avg_bar = alt.Chart(df_zone).mark_bar().encode(
    x = alt.X('BZone:N'),
    y = alt.Y('mean(reading_count):Q', title='Day Average'),
    tooltip=['BZone','mean(reading_count):Q'],
    color=alt.condition(zone_selector, 'BZone:N', alt.value('lightgray'), legend=None),
).transform_filter(
        time_brush
).add_selection(zone_selector)


#https://www.districtdatalabs.com/altair-choropleth-viz


(lat_lon_points | (month_avg_bar | day_avg_bar)) & (region_by_month)
# -





















# ### Anomaly Over Time

# +
# https://altair-viz.github.io/user_guide/interactions.html
# -

#Methane
ven_red = '#C91414'
cad_ed ='#E3071D'
amber ='#FF7E01'
flu_orange ='#FFBE00'
bud_green ='#75AD6F'
dark_green ='#1D7044'


# +
import boto3
import pickle

bucket = 'methane-capstone'
fdf_path = 'models/autoencoder/zone_model_artifacts/final_dataframes.pickle'

#Connect to S3 default profile
s3client = boto3.client('s3')

final_df_dict = pickle.loads(s3client.get_object(Bucket=bucket, Key=fdf_path)['Body'].read())
final_df_dict.keys()
# -

anom_df = pd.DataFrame()

# +
anom_cols = [x for x in final_df_dict[1]['train'].columns if 'thresh' not in x and 'loss' not in x]

fin_anom_cols = ['zone'] + anom_cols
fin_anom_cols = ['zone', 'methane_mixing_ratio_bias_corrected_mean', 'methane_mixing_ratio_bias_corrected_mean_anomaly']

anom_df = pd.DataFrame(columns=fin_anom_cols)
anom_df
# -

for zone in final_df_dict:
    for split in ['train', 'val','test']:
        
        cur_df = final_df_dict[zone][split]
        cur_df.insert(1, 'zone', [zone]*len(cur_df))
        cur_df = cur_df[fin_anom_cols]
        
        anom_df = pd.concat([anom_df, cur_df])

anom_df = anom_df.reset_index()
anom_df = anom_df.rename({'index': 'time_utc'}, axis= 1)
anom_df['time_utc'] = pd.to_datetime(anom_df['time_utc'])

anom_df = anom_df.rename({'methane_mixing_ratio_bias_corrected_mean_anomaly': 'anomaly', 
                           'methane_mixing_ratio_bias_corrected_mean': 'methane'
                          }, axis= 1)
anom_df.head()

g_anom_df = anom_df.set_index("time_utc").groupby([pd.Grouper(freq="M"), 'zone', 'anomaly']).size().reset_index().rename({0: 'count'}, axis =1)
g_anom_df['time_utc'] = pd.to_datetime(g_anom_df['time_utc'])
g_anom_df

g_anom_df["time_utc"] = pd.to_datetime(g_anom_df["time_utc"]).dt.strftime("%Y-%m")
g_anom_df['month_num'] = np.arange(1, len(g_anom_df) +1)
g_anom_df

# +
data_start = g_anom_df["time_utc"].min()
data_end = g_anom_df["time_utc"].max()


month_start = g_anom_df["month_num"].min()
month_end = g_anom_df["month_num"].max()


# +
# df["date2"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
# data_start = df["date"].min()
# data_end = df["date"].max()

# range_start = alt.binding(input="date")
# range_end = alt.binding(input="date")
# select_range_start = alt.selection_single(name="select_range_start", fields=["date"], bind=range_start, init={"date": data_start})
# select_range_end = alt.selection_single(name="select_range_end", fields=["date"], bind=range_end, init={"date": data_end})

# alt.Chart(df).transform_filter(
#   (datum.date2 >= select_range_start.date) & (datum.date2 <= select_range_end.date)
# ).add_selection(select_range_start, select_range_end).mark_line().encode(
#   x="date:T",
#   y="sum(views):Q",
#   color="ISO:N"
# )
# -

g_anom_df['count'].max()

g_anom_df.columns

# +
anom_bool = alt.Scale(domain=('Not Anomaly', 'Anomaly'),
                      range=[dark_green, ven_red])

slider = alt.binding_range(min=month_start, max=month_end, step=1)
select_year_month = alt.selection_single(name="year_month", fields=['month_num'],
                                   bind=slider, init={'month_num': 1})

alt.Chart(g_anom_df).mark_bar().encode(
    x=alt.X('anomaly:N', title=None),
    y=alt.Y('count:Q', scale=alt.Scale(domain=(0, 20000))),
    color=alt.Color('anomaly:N', scale=anom_bool),
    column='zone:O'
).properties(
    width=20
).add_selection(
    select_year_month
).transform_calculate(
    "anomaly", alt.expr.if_(alt.datum.anomaly, "Not Anomaly", "Anomaly")
).transform_filter(
    select_year_month
).configure_facet(
    spacing=8
)
# -






