# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Goal:
# Illustrate some examples of plotting our data using both MatplotLib and Altair. Using `august_filtered.csv`

# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from descartes import PolygonPatch
import altair as alt
import matplotlib.pyplot as plt 

import json
import geojson

#Altair puts a limit on plotting only 5000 rows from a pd.DataFrame. This line gets rid of that limit
alt.data_transformers.disable_max_rows()


# %% [markdown]
# ### Load Data

# %%
df = pd.read_csv('aug_21_filtered.csv').drop('Unnamed: 0', axis=1)

# Convert it to a GeoDataFrame by transforming the Latitude/Longitude coordinates 
loc_crs = {'init': 'epsg:4326'}

# Beware it may be good here to filter on a daily basis before running below lines of code
loc_geom = [Point(xy) for xy in zip(df['lon'], df['lat'])]
geo_df = gpd.GeoDataFrame(df, crs=loc_crs, geometry=loc_geom)
geo_df.head()

# %% [markdown]
# ### Basic plot, just points

# %%
geo_df.plot()

# %% [markdown]
# ### Load in GeoJSON file, create Polygon

# %%
geo_json_path = "../../data_processing/resources/california.geojson"
with open(geo_json_path) as json_file:
    geojson_data = geojson.load(json_file)

ca_poly = geojson_data['geometry']


# %% [markdown]
# ### Plot GeoJSON

# %%
BLUE = '#6699cc'
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.gca() 
ax.add_patch(PolygonPatch(ca_poly, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2 ))
ax.axis('scaled')
plt.show()

# %% [markdown]
# ### Plot points on top of map

# %%
fig, ax = plt.subplots(figsize=(10,10))

color = 'lightgrey' #'#6699cc'
#CA Basemap - Toggle zorder=0 and zorder=2 to move the california shape in front of points
ax.add_patch(PolygonPatch(ca_poly, fc=color, ec=color, alpha=0.5, zorder=1 )) 
ax.axis('scaled')

# Overlay the data locations - choose color map - https://matplotlib.org/stable/tutorials/colors/colormaps.html
# reverse color map by just adding '_r' to end of the string
geo_df.plot(ax=ax, alpha=0.5, zorder=2, cmap='plasma_r')

plt.show()

# %% [markdown]
# <hr>
#
# ### Using Altair
#
# * https://towardsdatascience.com/create-stunning-visualizations-with-altair-f9af7ad5e9b

# %%
#Total Data Shape
df_trim = df.iloc[:5000, 0:7]
print("All Data: ", df.shape)
print("Trimmed Data: ", df_trim.shape)

# %% [markdown]
# ### Create a Basemap of CA. We will overlap the scatter plots on top of this.

# %%
#Load CA GeoJSON
gdf = gpd.read_file(geo_json_path)
choro_json = json.loads(gdf.to_json())
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
ca_base

# %% [markdown]
# ### Example w/color on `qa_val`

# %%
#Plot all the readings
points = alt.Chart(df_trim).mark_circle().encode(
    longitude='lon:Q',
    latitude='lat:Q',
    tooltip= list(df_trim.columns),
    color='qa_val:Q'
)
ca_base + points

# %% [markdown]
# ### Example w/color on `methane_mixing_ratio`

# %%
#Plot all the readings
points = alt.Chart(df_trim).mark_circle().encode(
    longitude='lon:Q',
    latitude='lat:Q',
    tooltip= list(df_trim.columns),
    color=alt.Color('methane_mixing_ratio', scale=alt.Scale(scheme='viridis'))
)
ca_base + points

# %% [markdown]
# ### Example w/ color on `lon`

# %%
#Plot all the readings
points = alt.Chart(df_trim).mark_circle().encode(
    longitude='lon:Q',
    latitude='lat:Q',
    tooltip= list(df_trim.columns),
    color=alt.Color('lon', scale=alt.Scale(scheme='viridis')),
)
ca_base + points

# %%

# %% [markdown]
# ### Example w/scrub on `methane_mixing_ratio`

# %%
slider = alt.binding_range(min=int(min(df_trim['methane_mixing_ratio'])), max=int(max(df_trim['methane_mixing_ratio'])), step=1, name='cutoff:')
selector = alt.selection_single(name="SelectorName", fields=['cutoff'],
                                bind=slider, init={'cutoff': int(np.mean(df_trim['methane_mixing_ratio']))})

#Plot all the readings
points = alt.Chart(df_trim).mark_circle().encode(
    longitude='lon:Q',
    latitude='lat:Q',
    tooltip= list(df_trim.columns),    
    color=alt.condition(
        alt.datum.methane_mixing_ratio < selector.cutoff,
        alt.value('red'), alt.value('blue')
    )
).add_selection(
    selector
)


ca_base + points



# %%
