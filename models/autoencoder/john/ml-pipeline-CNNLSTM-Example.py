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

# ## Developer Name: -- John--
# ## Date: --10/23/2021--

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

frequency = '1D' 
round_level = 'rn_2'

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
keep_cols = ['time_utc', lat_str, lon_str, 'reading_count', 
             'methane_mixing_ratio_bias_corrected_mean', 
             dist_away_str, 'qa_val_mean']

df = df[keep_cols]
df.head()
# -

# ### Understand number of readings per time unit to figure out what padding should be, 
#
# * or set a limit of number of readings we want to consider

# +
df_readings_per_time_unit = df.groupby('time_utc').size().reset_index().rename({0:'reading_count'}, axis=1)
print(df_readings_per_time_unit.shape)

max_reading_per_day = int(df_readings_per_time_unit['reading_count'].max())
mean_reading_per_day = int(df_readings_per_time_unit['reading_count'].mean())
median_reading_per_day = int(df_readings_per_time_unit['reading_count'].median())

#Always pad up
rounded_max_reading_per_day = round(max_reading_per_day/10)*10 if round(max_reading_per_day/10)*10 > max_reading_per_day else (round(max_reading_per_day/10)*10) + 10

print()
print("max_reading_per_day:", max_reading_per_day)
print("mean_reading_per_day:", mean_reading_per_day)
print("median_reading_per_day:", median_reading_per_day)
print("rounded_max_reading_per_day:", rounded_max_reading_per_day)
print()

alt.Chart(df_readings_per_time_unit[['reading_count']]).mark_bar(tooltip=True).encode(
    alt.X("reading_count:Q", bin=True),
    y='count()',
).interactive()
# -
# ## Train / Test Split

# Enter a cutoff date
cutoff = datetime(2021, 1, 1) # Using 2021 as test data
train = df[df.time_utc < cutoff]
test = df[df.time_utc >= cutoff]

print(train.shape)
train.head()

print(test.shape)
test.head()

# ## Pre-Modeling Work

# Below code cells in this section describe examples of pre-modeling work to be done.
#
# **BE CAREFUL NOT TO STANDARD SCALER TWICE**

# #### Apply Standard Scalar

# +
scaler_mmrbc = StandardScaler()
scaler_mmrbc = scaler_mmrbc.fit(train[['methane_mixing_ratio_bias_corrected_mean']])

scaler_rc = StandardScaler()
scaler_rc = scaler_rc.fit(train[['reading_count']])

scaler_da = StandardScaler()
scaler_da = scaler_da.fit(train[[dist_away_str]])

scaler_qa = StandardScaler()
scaler_qa = scaler_qa.fit(train[['qa_val_mean']])


train['methane_mixing_ratio_bias_corrected_mean'] = scaler_mmrbc.transform(train[['methane_mixing_ratio_bias_corrected_mean']])
train['reading_count'] = scaler_rc.transform(train[['reading_count']])
train[dist_away_str] = scaler_da.transform(train[[dist_away_str]])
train['qa_val_mean'] = scaler_qa.transform(train[['qa_val_mean']])

test['methane_mixing_ratio_bias_corrected_mean'] = scaler_mmrbc.transform(test[['methane_mixing_ratio_bias_corrected_mean']])
test['reading_count'] = scaler_rc.transform(test[['reading_count']])
test[dist_away_str] = scaler_da.transform(test[[dist_away_str]])
test['qa_val_mean'] = scaler_qa.transform(test[['qa_val_mean']])

train.head()
# -

# #### A key step is to understand that based on the time granularity, `trainX`, `trainY` and `testX` and `testY` need to make sense

print('Print # of time steps in train', train['time_utc'].nunique())
print()
print('Print # of time steps in test', test['time_utc'].nunique())

# +
"""
Create windows for LSTM with all the data
"""
seq_window = 7
# x = train[keep_cols]
# y = train['methane_mixing_ratio_bias_corrected_mean']
# seq_size = seq_window

print('rounded_max_reading_per_day:', rounded_max_reading_per_day)
print()

def create_sequences_advanced(x, y, seq_size):

    start= time.time()
    assert 'time_utc' in set(x.columns), "time_utc column not present"

    time_steps = sorted(list(set(x.time_utc.tolist())))
    print()
    print('total timesteps', len(time_steps))

    x_values = []
    y_values = []

    for i in range(len(time_steps)-seq_size):

    #    cur_time_step = time_steps[i]
    #    print(cur_time_step)

        x_values_cur_ts_window = []
        for ts in time_steps[i:i+seq_size]:
            cur_day_df = x[x['time_utc'] == ts]

            #Readings
            cur_day_data = tf.cast(tf.constant(cur_day_df.iloc[:, 1:].values), tf.float64)

            #print("cur_day_data.shape", cur_day_data.shape)

            #Paddings
            pad_row, pad_col  = cur_day_data.shape[0], cur_day_data.shape[1]
            #print('pad_row', pad_row)
            #print('pad_col', pad_col)
            pad = tf.zeros(
                [rounded_max_reading_per_day-pad_row, pad_col], dtype=tf.float64
            )

            #print("pad.shape", pad.shape)
            #print()

            cur_day_x = tf.concat([cur_day_data, pad], 0)
            x_values_cur_ts_window.append(cur_day_x)


        x_values.append(x_values_cur_ts_window)
        y_values.append(y.iloc[i+seq_size])

    end= time.time()
    print(end-start, '\n')

    x_result = tf.expand_dims(x_values, 0)[0]
    y_result = tf.constant(y_values)

    #print('x_result.shape', x_result.shape)
    #print('y_result.shape', y_result.shape)

    return x_result, y_result



trainX, trainY = create_sequences_advanced(train[keep_cols],
                                  train['methane_mixing_ratio_bias_corrected_mean'],
                                  seq_window)

testX, testY = create_sequences_advanced(test[keep_cols], test['methane_mixing_ratio_bias_corrected_mean'], seq_window)

print("trainX Shape: ", trainX.shape, "trainY Shape: ", trainY.shape)
print("testX Shape: ", testX.shape, "testY Shape: ", testY.shape)
# -
#
# Phase 1:
#
#     * convert (790, 6) to a 1-D vector (SOME_D = 128) --> (790, 128)
#         * max, avg pooling some dense layers 
#         * dense layers to represent lat/lon
#         * embedding for lat/lon
#         * distance attention - after computing haversine distance
#             * potentially get rid of lat/lon, node (790,4)
#             * 790, 790 will store distance between points.
#             * https://www.kaggle.com/c/stanford-covid-vaccine/code?competitionId=22111&searchQuery=distance+attention
#             * https://www.kaggle.com/akashsuper2000/autoencoder-pretrain-gnn-attn-cnn
#         
#     * X: (750, 3 , SOME_D), Y:(METHANE)
#
# Phase 2:
#
# LSTM:
#
#     * take out time step if not bi-directional 
#     * keep the long sequences (all time step info)
#     * sequence to sequence is possible
#     
# Input: (750, 3, 790, 6)
# Output: 1-shifted right, (750, 3, 790, 6)
#     * no data leakage without windowing







# ### Questions
#
# * is this the best way to represent the time series data?
# * should it be batched any different by location chunks?
# * Structure wise I am trying to represent temporal and spatial features, does an embedding layer make sense for the spatial part? Any thoughts here?

# +
# """
# Create windows for LSTM
#     - As required by LSTM networks, we require to reshape an input data into 'n_samples' x 'timesteps' x 'n_features'.
#     - Number of time steps to look back. Larger sequences (look further back) may improve forecasting.
# """
# seq_size = 3

# def create_sequences_simple(x, y, seq_size=1):
#     x_values = []
#     y_values = []

#     for i in range(len(x)-seq_size):
#         x_values.append(x.iloc[i:(i+seq_size)].values)
#         y_values.append(y.iloc[i+seq_size])
        
#     return np.array(x_values), np.array(y_values)

# trainX, trainY = create_sequences_simple(train[['methane_mixing_ratio_bias_corrected_mean']], train['methane_mixing_ratio_bias_corrected_mean'], seq_size)
# testX, testY = create_sequences_simple(test[['methane_mixing_ratio_bias_corrected_mean']], test['methane_mixing_ratio_bias_corrected_mean'], seq_size)

# print("trainX Shape: ", trainX.shape, "trainY Shape: ", trainY.shape)
# print("testX Shape: ", testX.shape, "testY Shape: ", testY.shape)
# -

tuple(trainX.shape)

# ## Spatial Embedding Section
#
# A section to work through building out the spatial embeddings representation to be inputted as a layer for the model.

# +
print(df['rn_lat_2'].nunique())
print(df['rn_lon_2'].nunique())

print(train['rn_lat_2'].nunique())
print(train['rn_lon_2'].nunique())

print(test['rn_lat_2'].nunique())
print(test['rn_lon_2'].nunique())
# -

print(sorted(df['rn_lat_2'].unique().tolist()))
# print(df['rn_lon_2'].nunique())

# +
# # GEOGRAPHIC EMBEDDINGS CODE STARTS HERE
# # create dictionary of each lat/lon pair
# lats = np.array(sorted(df[lat_str].unique().tolist()))
# lons = np.array(sorted(df[lon_str].unique().tolist()))
# pairs = []

# for lat in lats:
#     for lon in lons:
#         pairs.append((round(lat, 1),round(lon,1)))
        
# print(len(pairs))

# # create haversine calculation
# # these calcs take too long for 0.2, 0.1 resolutions. Can filter by pairs that are present in the data

# def haversine_distance(point_a, point_b, unit = 'km'):

#     lat_s, lon_s = point_a[0], point_a[1] #Source
#     lat_d, lon_d = point_b[0], point_b[1] #Destination
#     radius = 6371 if unit == 'km' else 3956 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.

#     dlat = np.radians(lat_d - lat_s)
#     dlon = np.radians(lon_d - lon_s)
#     a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat_s)) * np.cos(np.radians(lat_d)) * np.sin(dlon/2) * np.sin(dlon/2)
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#     distance = radius * c
#     return distance

# # create matrix of distances (equivalent to co-occurence matrix for word embeddings)
# # these calcs take too long for 0.2, 0.1 resolutions. Can filter by pairs that are present in the data

# start=time.time()
# n = len(pairs)
# distances = np.zeros((n,n))

# for a in range(n):
#     for b in range(n):
#         distances[a][b] = haversine_distance(pairs[a], pairs[b])

# end=time.time()
# print(distances.shape, end-start)

# #SVD
# from sklearn.decomposition import TruncatedSVD
# def SVD(X, d=100):
#     """Returns word vectors from SVD.
    
#     Args:
#       X: m x n matrix
#       d: word vector dimension
      
#     Returns:
#       Wv : m x d matrix, each row is a word vector.
#     """
#     transformer = TruncatedSVD(n_components=d, random_state=1)
#     Wv = transformer.fit_transform(X)
#     # Normalize to unit length
#     Wv = Wv / np.linalg.norm(Wv, axis=1).reshape([-1,1])
#     return Wv, transformer.explained_variance_


# # # +
# # get final embeddings

# d = 10 # embedding dimension
# embeddings = SVD(distances, d=d)[0]


# # create dict w/ key = lat/lon pair, value = embedding vector

# emb_dict = {}

# for i in range(len(pairs)):
#     emb_dict[pairs[i]] = embeddings[i]
    
# # print(emb_dict)

# # # +
# # test out how it worked on a few points

# from scipy.spatial import distance

# point_a = emb_dict[(32.2, -120.4)]
# point_b = emb_dict[(32.2, -121.4)] # very close to point_a
# point_c = emb_dict[(32.2, -115.4)] # further from point_a
# point_d = emb_dict[(42.2, -115.4)] # very far from point_a

# print(f'between points a and b: {distance.euclidean(point_a, point_b)}')
# print(f'between points a and c: {distance.euclidean(point_a, point_c)}')
# print(f'between points a and d: {distance.euclidean(point_a, point_d)}')
# # -


# -

print('(number of training examples, time sequence window, number of rows per day, number of features)')
print()
print(trainX.shape)

trainX.shape

trainY.shape
#shape here should be (746, 790, 6)

# +
#KARTHIKS VERSION

# #Input shape
# inputs = keras.Input(shape=trainX.shape[1:])


# ### ### Neural Network Begins ### ### 
# dense_1 = layers.Dense(128, activation='relu')
# x = dense_1(inputs)

# dense_2 = layers.Dense(128, activation='relu')
# x = dense_2(x)

# avg_pool = layers.TimeDistributed(layers.GlobalAveragePooling1D()) #Here we can do flatten as well,
# # max_pool = layers.MaxPooling1D(pool_size=2)
# #Weight average of max vs. avg. ---> GEM pool

# output_avg = avg_pool(x)
# # output_max = max_pool(x)
# # layers.Concatenate() to join the maxpool and average pool 
# # layer_3 = layers.Flatten()

# # Create the layers of the model
# lstm_enc_1 = keras.layers.LSTM(units=64,)(output_avg)
# # lstm_enc_2 = keras.layers.LSTM(units=128)(lstm_enc_1)

# drop_1 = keras.layers.Dropout(rate=0.2)(lstm_enc_1)
# repeat = keras.layers.RepeatVector(n=trainX.shape[1])(drop_1)
# lstm_dec_2 = keras.layers.LSTM(units=128, return_sequences=True)(repeat)
# drop_2 = keras.layers.Dropout(rate=0.2)(lstm_dec_2)
# dense_3 = keras.layers.Dense(units=4740)(drop_2)
# # dense_4 = keras.layers.Dense(units=1)(dense_3)
# dense_4 = keras.layers.Reshape((7, 790,6))(dense_3)
# outputs = dense_4

# # layers.Reshape((28, 28))

# model = keras.Model(inputs=inputs, outputs=outputs, name="methane_spatio_temporal")
# model.summary()
# -

# * Use auto-encoder pre-training to learn back the input vector
# * with distance attention 

# # JOHN STARTED HERE

from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input
# MISSING BATCH NORMALIZATION B/C I COULDN"T GET THE PACKAGE TO WORK
# from keras.layers.normalization import BatchNormalization


trainX.shape

# + jupyter={"outputs_hidden": true}
# HAD TO RESHAPE TO GET THE CNN CODE TO WORK.

"""
REMEMBER:
746 or 264 = days in training data or testing    *note that model has parameter of 0.2 for validation data*
7 = sequence (7 days per input meaning there should be 739 or 740 input data and not 746)
790 = regions based on unique combinations of lat/lon 
1 = just an extra dimension for the 790 so that the CNN would work
6 = 6 regressors,  4th or index-3 should be methane readings

"""

trainX2 = tf.reshape(trainX, (746, 7, 790, 1, 6))
print(trainX2.shape)
testX2 = tf.reshape(testX, (264,7,790,1,6))
print(testX2.shape)

# +
#JOHN's VERSION CNNLSTM  #1 with bad results

#Input shape

inputs = keras.Input(shape=trainX2.shape[1:])


# ### Neural Network Begins ###
# MISSING BATCH NORMALIZATION B/C I COULDN"T GET THE PACKAGE TO WORK

# encode
conv1 = TimeDistributed(Conv2D(128, kernel_size=(4, 1), padding='same', strides=(10, 1), name='conv1'),
                        input_shape=trainX2.shape[1:])(inputs)   #CHOOSE STRIDES THAT ARE FACTORS OF TRAINING DATA SHAPE TO OUTPUT CORRECT SHAPE
conv1 = TimeDistributed(Activation('relu'))(conv1)


convlstm1 = ConvLSTM2D(64, kernel_size=(4, 1), padding='same', return_sequences=True, name='convlstm1')(conv1)

# decode
deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(4, 1), padding='same', strides=(10, 1), name='deconv1'))(convlstm1)
deconv1 = TimeDistributed(Activation('relu'))(deconv1)


decoded = TimeDistributed(Conv2DTranspose(6, kernel_size=(4, 1), padding='same', strides=(1, 1), name='deconv2'))(
    deconv1)


outputs=decoded

model = keras.Model(inputs=inputs, outputs=outputs, name="methane_spatio_temporal")
model.summary()
# -

# Compile the model
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

#Fit model
# TAKES 13 MINUTES TO TRAIN ON ml.m5.large
# NOTE THAT 'trainX2' is both input and output.  We are not forecasting, we are reconstructing the 7 day windows
mod1 = model.fit(
    trainX2, trainX2,
    batch_size=32,
    epochs = 50,
    validation_split = 0.2,
    shuffle = False,
)

# plotting
plt.plot(mod1.history['loss'])
plt.plot(mod1.history['val_loss'])
plt.title("mod1 Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()

# +
#JOHN's VERSION CNNLSTM  MODEL 2 with MORE LAYERS

#Input shape

inputs = keras.Input(shape=trainX2.shape[1:])


# ### Neural Network Begins ###
# MISSING BATCH NORMALIZATION B/C I COULDN"T GET THE PACKAGE TO WORK

#encode
conv1 = TimeDistributed(Conv2D(128, kernel_size=(4, 1), padding='same', strides=(5, 1), name='conv1'),
                        input_shape=trainX2.shape[1:])(inputs) #CHOOSE STRIDES THAT ARE FACTORS OF TRAINING DATA SHAPE TO OUTPUT CORRECT SHAPE
conv1 = TimeDistributed(Activation('relu'))(conv1)
conv2 = TimeDistributed(Conv2D(128, kernel_size=(4, 1), padding='same', strides=(2, 1), name='conv1'),
                        input_shape=trainX2.shape[1:])(conv1)
conv2 = TimeDistributed(Activation('relu'))(conv2)


convlstm1 = ConvLSTM2D(64, kernel_size=(4, 1), padding='same', return_sequences=True, name='convlstm1')(conv2)

#decode
deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(4, 1), padding='same', strides=(2, 1), name='deconv1'))(convlstm1)
deconv1 = TimeDistributed(Activation('relu'))(deconv1)
deconv2 = TimeDistributed(Conv2DTranspose(128, kernel_size=(4, 1), padding='same', strides=(5, 1), name='deconv1'))(deconv1)
deconv2 = TimeDistributed(Activation('relu'))(deconv2)


decoded = TimeDistributed(Conv2DTranspose(6, kernel_size=(4, 1), padding='same', strides=(1, 1), name='deconv2'))(
    deconv2)


outputs=decoded

model = keras.Model(inputs=inputs, outputs=outputs, name="methane_spatio_temporal")
model.summary()

# +
# Compile the model
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

#Fit model
# TAKES 13 MINUTES TO TRAIN ON ml.m5.large
# NOTE THAT 'trainX2' is both input and output.  We are not forecasting, we are reconstructing the 7 day windows
mod1 = model.fit(
    trainX2, trainX2,
    batch_size=32,
    epochs = 50,
    validation_split = 0.2,
    shuffle = False,
)

# +
#plotting

plt.plot(mod1.history['loss'])
plt.plot(mod1.history['val_loss'])
plt.title("mod1 Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()
# -


# ### GETTING READY TO CREATE ERROR DATAFRAME AND ANOMALY PREDICTION
#

# +
#Predict on test
test_preds = model.predict(testX2)
print(test_preds.shape)

#Predict on train
train_preds = model.predict(trainX2)
print(train_preds.shape)
# -

#NEED TO ADJUST FOR METHANE DATA ONLY AND NOT ALL 6 REGRESSORS
train_mae_loss  = np.mean(np.abs(train_preds - trainX2), axis=1)
print(train_mae_loss.shape)

#NEED TO ADJUST FOR METHANE DATA ONLY AND NOT ALL 6 REGRESSORS
test_mae_loss  = np.mean(np.abs(test_preds - testX2), axis=1)
test_mae_loss.shape

# 3 is the methane data, ignoring the other regressors
pd.DataFrame(train_mae_loss[:,:,0,3])

# Used Test dataframe to get the time_utc for indexing the anomaly dataframe
test.head()

# +
# CREATE TABLE OF TEST_SCORE_DF
# THE ERROR FOR EACH LAT/LON COMBO AT EACH START DATE (WE ALREADY AVERAGED THE 7 DAY SEQUENCE ERROR ALREADY)

upper,lower = np.percentile(pd.DataFrame(train_mae_loss[:,:,0,3]), [75,25])     #get the interquartile range (75% and 25%) of entire test_mae_loss
anom_thresh = 1.5*(upper-lower)
seq_size =7 
test_score_df = pd.DataFrame(test_mae_loss[:,:,0,3])  #create dataframe from test_mae_loss, each column is lat/lon combo, each row is date
test_score_df.index = [str(x).split('T')[0] for x in list(test['time_utc'].unique()[:264])]   #add index to show that each row is a date
# test_score_df.columns = [name + ' loss' for name in test.columns]  #add column labels (just the lat/lon combos)
test_score_df['threshold'] = anom_thresh
for col in test_score_df.columns:
    test_score_df[str(col) + ' anomaly'] = test_score_df[col] > test_score_df.threshold   #add anomaly column for each lat/lon combo
test_score_df
# test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[seq_size:].methane_mixing_ratio_mean

# -

anomaly_columns = test_score_df.columns[791:-1]  #get the anomaly columns to count how many are true
anomaly_columns

# +
anomaly_columns = test_score_df.columns[791:-1]  #get the anomaly columns to count how many are true
anomaly_df = pd.DataFrame(index=anomaly_columns)  #create dataframe where index is each lat/lon

#add anomaly count and anomaly index    
anomaly_index_list = []
anomaly_count_list = []
for anomaly_col in anomaly_columns:
    anomaly_index = test_score_df.index[test_score_df[anomaly_col]==True]
    anomaly_index_list.append( anomaly_index )
    anomaly_count_list.append( len(anomaly_index) )
    
anomaly_df['anomaly_count'] = anomaly_count_list
anomaly_df['anomaly_index'] = anomaly_index_list
anomaly_df
# -

plt.hist(anomaly_df['anomaly_count'])






