# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (TensorFlow 2.3 Python 3.7 CPU Optimized)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/tensorflow-2.3-cpu-py37-ubuntu18.04-v1
# ---

# # ML Pipeline Template

# This notebook is to be referenced as a template for the ML pipelines we build for the autoencoder framework.

# ## Developer Name: Jaclyn
# ## Date: 10/23/2021

# ## Imports

# The following libraries have already been installed due to the AWS Lifecycle Configuration:
# - pandas 
# - numpy 
# - geopandas 
# - altair 
# - geojson 
# - matplotlib 
# - plotly 
# - descartes 
# - tensorflow 
# - keras 
# - tensorboard 
# - seaborn 
# - xarray 
# - jupytext
#
# Add all necessary imports here.

# +
# Downgrade Tensorflow to version 2.3
# # !pip install tensorflow==2.3

# +
import os
import altair as alt
import boto3
from datetime import datetime
import time
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# -

# Important to ensure we are using Tensorflow version 2.3
print(tf.__version__)
tf.executing_eagerly()

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 18

# ## Paths

ROOT_DIR = '/root/methane/models'
bucket = 'methane-capstone'
subfolder = 'data/data-variants'
s3_path = bucket+'/'+subfolder
print('s3_path: ', s3_path)

# ## Load Data

# +
#CHOOSE CONFIG

frequency = '3D' 
round_level = 'rn_1'


if round_level == 'rn':
    res = 1.0
elif round_level == 'rn_5':
    res = 0.5
elif round_level == 'rn_2':
    res = 0.2
elif round_level == 'rn_1':
    res = 0.1
else:
    None

#### #### #### #### ####

lat_str=''
lon_str=''
dist_away_str=''

if '1' in round_level:
    lat_str = 'rn_lat_1'
    lon_str = 'rn_lon_1'
    dist_away_str = 'dist_away_1_mean'
elif '2' in round_level:
    lat_str = 'rn_lat_2'
    lon_str = 'rn_lon_2'
    dist_away_str = 'dist_away_2_mean'
elif '5' in round_level:
    lat_str = 'rn_lat_5'
    lon_str = 'rn_lon_5'
    dist_away_str = 'dist_away_5_mean'

else:
    lat_str = 'rn_lat'
    lon_str = 'rn_lon'
    dist_away_str = 'dist_away_mean'


file_name=f'data_{frequency}_{round_level}.parquet.gzip' # Insert specific data variant file name here
data_location = 's3://{}/{}'.format(s3_path, file_name)
df = pd.read_parquet(data_location)
df['time_utc'] = pd.to_datetime(df['time_utc'])
print(df.shape)
df.head()
# -

# Define columns we want to use

# +
### ENSURE THAT `time_utc` is the value in the 0th index!!
# keep_cols = ['time_utc', lat_str, lon_str, 'reading_count', 
#              'methane_mixing_ratio_bias_corrected_mean', 
#              dist_away_str, 'qa_val_mean']

keep_cols = ['time_utc', lat_str, lon_str, 
             'methane_mixing_ratio_bias_corrected_mean']

df = df[keep_cols]
df_time = df.set_index('time_utc')
df_time.head()
# -

# ### Understand number of readings per time unit to figure out what padding should be, 
#
# * or set a limit of number of readings we want to consider

# +
# df_readings_per_time_unit = df_time.groupby('time_utc').size().reset_index().rename({0:'reading_count'}, axis=1)
# print(df_readings_per_time_unit.shape)

# max_reading_per_day = int(df_readings_per_time_unit['reading_count'].max())
# mean_reading_per_day = int(df_readings_per_time_unit['reading_count'].mean())
# median_reading_per_day = int(df_readings_per_time_unit['reading_count'].median())

# #Always pad up
# rounded_max_reading_per_day = round(max_reading_per_day/10)*10 if round(max_reading_per_day/10)*10 > max_reading_per_day else (round(max_reading_per_day/10)*10) + 10

# print()
# print("max_reading_per_day:", max_reading_per_day)
# print("mean_reading_per_day:", mean_reading_per_day)
# print("median_reading_per_day:", median_reading_per_day)
# print("rounded_max_reading_per_day:", rounded_max_reading_per_day)
# print()

# alt.Chart(df_readings_per_time_unit[['reading_count']]).mark_bar(tooltip=True).encode(
#     alt.X("reading_count:Q", bin=True),
#     y='count()',
# ).interactive()
# -
# ### Treat Lat/Lon as a categorical

# +
# number of unique latitude and longitude values
df_time[[lat_str, lon_str]].nunique()

max_lat = round(df_time[lat_str].max(),1)
max_lon = round(df_time[lon_str].max(),1)
min_lat = round(df_time[lat_str].min(),1)
min_lon = round(df_time[lon_str].min(),1)

print(f'{min_lat}, {max_lat}, {min_lon}, {max_lon}')
# -

df_time[[lat_str, lon_str]].nunique()

# +
# df_g_lat_lon = df.groupby([lat_str, lon_str]).size().reset_index()
# df_g_lat_lon = df_g_lat_lon[[lat_str, lon_str]].sort_values(by=[lat_str, lon_str], ascending=[False, True])
# print(df_g_lat_lon.shape)
# df_g_lat_lon.head()


# Jaclyn add --- include ALL lat/lon instead of just ones with measurements in order to make 'image'
all_lat = pd.DataFrame(np.arange(min_lat, max_lat+res, res), columns=[lat_str])
all_lon = pd.DataFrame(np.arange(min_lon, max_lon+res, res), columns=[lon_str])
df_g_lat_lon = all_lat.merge(all_lon, how='cross')
df_g_lat_lon[lat_str] = round(df_g_lat_lon[lat_str],1).astype(str)
df_g_lat_lon[lon_str] = round(df_g_lat_lon[lon_str],1).astype(str)
print(len(all_lat))
print(len(all_lon))
print(df_g_lat_lon.shape)
df_g_lat_lon.tail()
# -



# +
# get a dataframe of all times at the given frequency

start_dt = df_time.index.min()
end_dt = df_time.index.max()

print(start_dt)
print(end_dt)

time_table = pd.DataFrame(pd.date_range(start=start_dt, end=end_dt, freq=frequency), columns=['time_utc'])
print(time_table.shape)
time_table.head()
# -



# +
# all time, all latitudes and longitudes

all_time_lat_lon = time_table.merge(df_g_lat_lon, how='cross')
print(all_time_lat_lon.shape)
all_time_lat_lon.head()
# -






# +
# data frame without filling in missing times and lat/lons

df = df_time.reset_index()

print(df.shape)
df.head()

# +
# merge for final data set with missing lat/lon and times include as NaN
df[lat_str] = round(df[lat_str],1).astype(str)
df[lon_str] = round(df[lon_str],1).astype(str)
final_df = df.merge(all_time_lat_lon, on=['time_utc', lat_str, lon_str], how='outer')

print(final_df.shape)
final_df.head()
# -

# count of null methane values
final_df.isnull().sum()

# +
# standard scalar on methane value

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data=np.array(final_df['methane_mixing_ratio_bias_corrected_mean']).reshape(-1,1)
scaler.fit(data)
final_df['m_scaled'] = scaler.transform(data)
final_df[lat_str] = final_df[lat_str].astype(float)   # change back to float after changing to 
final_df[lon_str] = final_df[lon_str].astype(float)
# -

final_df.head()

# ### need to add code in here to create "images" for each time period

import matplotlib.pyplot as plt
def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))


# +
# loops through each time period and makes a dataframe with lat as rows, lon as columns -- add each to a list
# should take less than a minute regardless of resolution or time window

images = []
for time in final_df.time_utc.unique():
    one_day = final_df[final_df.time_utc == time].fillna(0)
    one_day_pivot = one_day.pivot(index=lat_str, columns=lon_str, values="m_scaled").sort_values(by=[lat_str], ascending=[False])
    images.append(one_day_pivot)
    
# -



# +
# extract and visualize one day

day = '2021-1-28'

one_day = final_df[pd.DatetimeIndex(final_df.time_utc).to_period('D') == day ]
one_day_pivot = one_day.pivot(index=lat_str, columns=lon_str, values="m_scaled")
one_day_pivot = one_day_pivot.sort_values(by=[lat_str], ascending=[False])
show_image(one_day_pivot)
# -

# takes all the "images" and stacks them --  I did not split to train/val 
X = np.stack(images).astype('float32')



# +
# from : https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/
# from tensorflow import keras 
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) 
    decoder.add(Reshape(img_shape))

    return encoder, decoder


# +
# we neglect the number of instances from shape
IMG_SHAPE = X.shape[1:]
encoder, decoder = build_autoencoder(IMG_SHAPE, 30) # lower code_size is more compression

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())
# -

history = autoencoder.fit(x=X, y=X, epochs=20,
                validation_data=[X, X])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# +
# show some examples of reconstructed images

def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""

    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

for i in range(5):
    img = X[i]
    visualize(img,encoder,decoder)
# -

img = X[300]
visualize(img,encoder,decoder)

# # END OF JACLYN WORK



# ## Train / Test Split

# +
final_df = final_df.set_index('time_utc')
train_date_threshold = '2021-01-01' 
validation_date_threshold = '2021-06-01'

train = final_df.loc[final_df.index < train_date_threshold]
validation = final_df.loc[(final_df.index >= train_date_threshold) & (final_df.index < validation_date_threshold)]
test = final_df.loc[final_df.index >= validation_date_threshold]

print(train.shape, validation.shape, test.shape)
# -

# ### Notes:
#
# * we might need some special treatment of the lat/lon, apply the embedding layer Jaclyn worked on here itself?
# * In https://project.inria.fr/aaltd19/files/2019/08/AALTD_19_Karadayi.pdf they treat the locations as categorical places, so we would need to do that 

train

# ### Todos:

# +
# feature_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count', dist_away_str, 'qa_val_mean']
feature_cols = ['methane_mixing_ratio_bias_corrected_mean']



train_input = train[feature_cols]
val_input = validation[feature_cols]
test_input = test[feature_cols]


scaler = MinMaxScaler() #StandardScaler()
scaler = scaler.fit(train[feature_cols])

train[feature_cols] = scaler.transform(train_input)
validation[feature_cols] = scaler.transform(val_input)
test[feature_cols] = scaler.transform(test_input)

# -

print('Print # of time steps in train', train.index.nunique())
print()
print('Print # of time steps in validation', validation.index.nunique())
print()
print('Print # of time steps in test', test.index.nunique())
print()
print("frequency", frequency)

train.head()

# ### I deleted the rest of the cells from Karthik's notebook because my notebook was running really slow :(


