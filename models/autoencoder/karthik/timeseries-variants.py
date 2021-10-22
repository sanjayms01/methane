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

# ## Goal 
#
# * Load in all of our data, and select a specific region that we want to model over time.
# * Build out an auto-encoder anomaly detection model that models this region

# +
# # !pip install geopandas altair geojson matplotlib plotly descartes
# -

# ### Imports

# +
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import altair as alt


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
# -

#Altair puts a limit on plotting only 5000 rows from a pd.DataFrame. This line gets rid of that limit
alt.data_transformers.disable_max_rows()

# ### Load all data

# +
#CHOOSE CONFIG

frequency = '1D' 
round_level = 'rn_2'
s3_path = 's3://methane-capstone/data/data-variants'

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

# ### Load in CA Base Map to Validate data on

# +
geo_json_path = "../data_processing/resources/california.geojson"
with open(geo_json_path) as json_file:
    geojson_data = geojson.load(json_file)
ca_poly = geojson_data['geometry']

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
    width=300,
    height=300
)
# -

# ## Geo Trimming
# * Either set your region
# * Or pick one exact point, and resolution
# * All readings will be averaged once per day regardles!
#
#
# For now, this is dodging the need to add spatial features to a model by restricting our data to a specific region
#
#

df_spot_check = df.groupby([lat_str, lon_str]).size().reset_index().rename({0:'count'}, axis=1).sort_values(by='count', ascending=False)
df_spot_check.head()

df_spot_check['count'].hist()

# +
rounded_lat, rounded_lon = df_spot_check[0:1:].values[0][:2] #OR pick a spot

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

print("SPOT round_level", round_level)

if '1' in round_level:
    geo_mask = (df.rn_lat_1 == rounded_lat) & \
               (df.rn_lon_1 == rounded_lon)
elif '2' in round_level:
    geo_mask = (df.rn_lat_2 == rounded_lat) & \
               (df.rn_lon_2 == rounded_lon)

elif '5' in round_level:
    geo_mask = (df.rn_lat_5 == rounded_lat) & \
               (df.rn_lon_5 == rounded_lon)
    
else:
    geo_mask = (df.rn_lat == rounded_lat) & \
               (df.rn_lon == rounded_lon)
    
keep_cols = ['time_utc', lat_str, lon_str, 'reading_count', 'methane_mixing_ratio_bias_corrected_mean', dist_away_str, 'qa_val_mean']
df_trim = df[geo_mask][keep_cols]

print(f"Losing {'{:,}'.format(df.shape[0] - df_trim.shape[0])} rows")
print()
print(f"df_trim: {df_trim.shape}")
print()
df_trim.head()
# -

# ### Validate Data is in the region you specified

#Plot all the readings
points = alt.Chart(df_trim).mark_circle(size=100).encode(
    longitude=f'{lon_str}:Q',
    latitude=f'{lat_str}:Q',
    tooltip= list(df_trim.columns)
)
ca_base + points

# +
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 10, 8

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator


# +
print(df_trim.shape)
print("start_dt:", df_trim['time_utc'].min(), "\nend_dt:", df_trim['time_utc'].max(), "\nnumber_days:", df_trim['time_utc'].max() - df_trim['time_utc'].min())
print()

df_trim = df_trim.set_index('time_utc')

df_trim.head()
# -

feature_cols = ['reading_count', 'methane_mixing_ratio_bias_corrected_mean', 'dist_away_2_mean', 'qa_val_mean']

plt.plot(df_trim['methane_mixing_ratio_bias_corrected_mean'], label='methane_mixing_ratio_bias_corrected_mean')

df_trim[feature_cols].plot(subplots=True)







# ### Train/Test Split 
# * by date 
# * by data size

# +
by_dt = True
date_threshold = '2021-01-01'
train_percent = 0.95

if by_dt:
    train, test = df_trim.loc[df_trim.index < date_threshold], df_trim.loc[df_trim.index >= date_threshold]
        
else:
    
    train_size = int(len(df_trim) * train_percent)
    test_size = len(df_trim) - train_size
    train, test = df_trim.iloc[:train_size], df_trim.iloc[train_size:]
    

print(train.shape, test.shape)
# -

train.head()

# ## Approaches:
#
# ### Single GeoLocation:
#
#    #### Univariate Time Series:
#    * https://towardsdatascience.com/time-series-of-price-anomaly-detection-with-lstm-11a12ba4f6d9
#    
#    * One time series feature --> [`methane_mixing_ratio_bias_corrected_mean`]
#         * Window over last 7 days --> predict on next day. 
#         * Autoencoder makes sense, because we are recreating the input. Shapes make sense.
#             * Input Shape (None, 7, 1)
#             * Output Shape (None, 7, 1)  <--- this is still trying to recreate the input
#
#    #### Multivariate Time Series:
#    
#    * Multiple time series features --> [`reading_count`,`methane_mixing_ratio_bias_corrected_mean`,`dist_away_2_mean`,`qa_val_mean`]
#         * https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1 
#         * https://colab.research.google.com/gist/onesamblack/fae25efdcd82ba208c453bafeba86d3c/lstm_autoencoder.ipynb
#         * Window over last 7 days --> predict on next day. 
#         * Autoencoder **not** making as much sense, because we are recreating the input. We don't want to recreate all the other features....
#             * Input Shape (None, 7, 4)
#             * Output Shape (None, 7, 1)  
#             * As of now shape of 1 works. Because Y variable is dimension 1. but doesn't make sense...w/ Autoencoder
#             * I would have to make last dense layer back to 4, make our Y variable also of dimension 4, but that is recreating all the features... not exactly what we want either
#
#    * Stacked Autoencoder
#         * This may be the architecture we need for this task.
#         
#
#
# ### Multiple GeoLocation:
#    #### ....
#
#
#
#
# ### Questions for Colorado:
#
# * Should we apply standard scalar to our feature vectur (a.k.a horizontally for every row) or on the whole column?
#     * Or do we take care of this with **batch normalization** in the NN  
#     * ANSWER: Maybe after first layer, btu try putting BN in a few places, maybe after each layer, try it before the dropout, 
#     * Variational Auto encoders
#     
#
# * How do we up sample back to the same input dimension that we initially aimed for, while maintaining time series windowing?    
# * Other Points:
#     * gotta store the scalar objets somewhere in memory at infer time
#

# ### Scale Values
# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset

# ### Create Windows for LSTM
#
# As required for LSTM networks, we require to reshape an input data into `batch` x `n_samples` x `timesteps` x `n_features`. 
#
#     * Organized via batches now with the generator
#     
# Time Series Generator Example/Documentation:
#
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
#     





# ### Univariate Model

# +
feature_cols = ['methane_mixing_ratio_bias_corrected_mean']

train_input = train[feature_cols]
test_input = test[feature_cols]

# s_scaler = StandardScaler()
# s_scaler = s_scaler.fit(train_input)

mm_scaler = MinMaxScaler()
mm_scaler = mm_scaler.fit(train_input)

train_scaled = mm_scaler.transform(train_input)
test_scaled = mm_scaler.transform(test_input)



# +
train_features = train_scaled
train_targets = train_scaled[:,0]

test_features = test_scaled
test_targets = test_scaled[:,0]

print(train_features.shape)
print(train_targets.shape)
print(test_features.shape)
print(test_targets.shape)

# -



# +
seq_size = 7
batch_size = 32
num_features = len(feature_cols)

train_gen = TimeseriesGenerator(train_features, train_targets, length=seq_size, batch_size=batch_size, shuffle=False)
test_gen = TimeseriesGenerator(test_features, test_targets, length=seq_size, batch_size=batch_size, shuffle=False)

print('Train batches:', len(train_gen))
print('Test batches:', len(test_gen))
print()
print('seq_size:', seq_size)
print('batch_size:', batch_size)
print('num_features:', num_features)

# -

train_gen[0][0].shape

train_gen[0][1].shape



# +
### Example of one batch of the data
print("X Shape")
print(train_gen[0][0].shape)

print("Y Shape")
print(train_gen[0][1].shape)
# -



# +
model = keras.Sequential()

#Here we can include other things like readings per day.
model.add(keras.layers.LSTM(units=64, input_shape = (seq_size, num_features)))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=seq_size))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_features)))

model.compile(loss=tf.losses.MeanAbsoluteError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanSquaredError(),tf.losses.MeanAbsoluteError()]
             )


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)




model.summary()
# -

# fit model
history = model.fit_generator(train_gen, 
                              epochs=100,
                              validation_data = test_gen,
                              shuffle=False,
                              callbacks=[early_stopping]
                             )


plt.title('MAE Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

# +
X_train_pred = model.predict_generator(train_gen)
print('predicted train shape:', X_train_pred.shape)

X_train = None
for ind,x in enumerate(train_gen, 1):
    if ind == 1:
        X_train = x[0]
    else:
        X_train = tf.concat([X_train, x[0]], axis=0)

X_train = X_train.numpy()        
print('original train shape:', X_train.shape)


train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)
print('train_mae_loss shape: ', train_mae_loss.shape)
# -

# error for each example from our training set
sns.distplot(train_mae_loss, bins=50, kde=True)

# +
X_test_pred = model.predict_generator(test_gen)
print('predicted test shape:', X_test_pred.shape)

X_test = None
for ind, x in enumerate(test_gen, 1):
    if ind == 1:
        X_test = x[0]
    else:
        X_test = tf.concat([X_test, x[0]], axis=0)

X_test = X_test.numpy()        
print('original test shape:', X_test.shape)


test_mae_loss = np.mean(np.abs(X_test_pred, X_test), axis=1)
print('test_mae_loss shape: ', test_mae_loss.shape)
# -

# error for each example from our training set
sns.distplot(test_mae_loss, bins=50, kde=True)

test

# +
ANOMALY_THRESHOLD = 0.9

test_score_df = pd.DataFrame(index=test[seq_size:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = ANOMALY_THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[seq_size:].methane_mixing_ratio_bias_corrected_mean
test_score_df['rn_lat_2'] = test[seq_size:].rn_lat_2
test_score_df['rn_lon_2'] = test[seq_size:].rn_lon_2
test_score_df['reading_count'] = test[seq_size:].reading_count
test_score_df['dist_away_2_mean'] = test[seq_size:].dist_away_2_mean 
test_score_df['qa_val_mean'] = test[seq_size:].qa_val_mean



plt.plot(test_score_df.index, test_score_df.loss, label = 'loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Test Loss vs. Anomaly Loss Threshold")
plt.legend()

# -

test_score_df.head()

# +
anomalies = test_score_df[test_score_df.anomaly]

plt.plot(
    test[seq_size:].index,
    mm_scaler.inverse_transform(test_scaled[seq_size:]),
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    anomalies.index,
    anomalies.methane_mixing_ratio_bias_corrected_mean,
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
# -

anomalies









# ### Multivariate Model - Forecasting.....
#

def generate_datasets_for_fc(data, window_size):
    _l = len(data) 
    Xs = []
    Ys = []
    for i in range(0, (_l - window_size)):
        # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        Xs.append(data[i:i+window_size])
        Ys.append(data[:,0][i+window_size])
        
    Xs=np.array(Xs)
    Ys=np.array(Ys)
    return  (Xs.shape[2], Xs, Ys)



# +
feature_cols = ['methane_mixing_ratio_bias_corrected_mean', 'reading_count','dist_away_2_mean', 'qa_val_mean']

train_input = train[feature_cols]
test_input = test[feature_cols]

# s_scaler = StandardScaler()
# s_scaler = s_scaler.fit(train_input)

mm_scaler = MinMaxScaler()
mm_scaler = mm_scaler.fit(train_input)

train_scaled = mm_scaler.transform(train_input)
test_scaled = mm_scaler.transform(test_input)



# +
# Sanity Check
# test_scaled[window_length:][:,0] == testY

# +
num_feats_train, trainX, trainY = generate_datasets_for_fc(train_scaled, 7)
num_feats_test, testX, testY = generate_datasets_for_fc(test_scaled, 7)

print(num_feats_train)
print(num_feats_test)
print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)
# -





# +
epochs = 100
batch_size = 32
window_length = 7


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)


model = keras.Sequential()
model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', batch_input_shape=(None, window_length, num_feats_train), return_sequences=True, name='encoder_1'))
model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'))
model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3'))
model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))
model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))

model.compile(loss=tf.losses.MeanAbsoluteError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanSquaredError(),tf.losses.MeanAbsoluteError()]
             )

model.build()
print(model.summary())

# -

history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=100, batch_size=batch_size, shuffle=False, callbacks=[early_stop])


plt.title('mae Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()

# +
print(num_feats_train)
print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)
# -

## CHECK trainX
trainX.shape

# #### Question!!! 
#
# * Can I do this????

## CHECK trainX
trainXMethane = trainX[:,:,:1]
trainXMethane.shape

# #### Question!!! 
#
# * I don't understand how after I run the below cell, `trainX` get's over written!!

# +


X_train_pred = model.predict(trainX)
print('predicted train shape:', X_train_pred.shape)       
print('original train shape:', trainXMethane.shape)

train_mae_loss = np.mean(np.abs(X_train_pred, trainXMethane), axis=1)
print('train_mae_loss shape: ', train_mae_loss.shape)
# -

# I'm not exactly sure how this is computed...??
sns.distplot(train_mae_loss, bins=50, kde=True)

## CHECK trainX
testXMethane = testX[:,:,:1]
testXMethane.shape

X_test_pred = model.predict(testX)
print('predicted test shape:', X_test_pred.shape)
print('original test shape:', testXMethane.shape)
test_mae_loss = np.mean(np.abs(X_test_pred, testXMethane), axis=1)
print('test_mae_loss shape: ', test_mae_loss.shape)

# error for each example from our training set
sns.distplot(test_mae_loss, bins=50, kde=True)

# #### Question!!! 
#
# * Here when I look at the loss, it is loss with respect to each of the 4 features, can I take an average of the loss and is that a valid choice?

# +
ANOMALY_THRESHOLD = 0.83

test_score_df = pd.DataFrame(index=test[window_length:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = ANOMALY_THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[window_length:].methane_mixing_ratio_bias_corrected_mean
test_score_df['rn_lat_2'] = test[window_length:].rn_lat_2
test_score_df['rn_lon_2'] = test[window_length:].rn_lon_2
test_score_df['reading_count'] = test[window_length:].reading_count
test_score_df['dist_away_2_mean'] = test[window_length:].dist_away_2_mean 
test_score_df['qa_val_mean'] = test[window_length:].qa_val_mean


plt.plot(test_score_df.index, test_score_df.loss, label = 'loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Test Loss vs. Anomaly Loss Threshold")
plt.legend()

# -

test_score_df.head()

# #### Question!!! 
#
# * Is this valid for me to inverse transform my scalar operation, just extract the 

just_methane_column = mm_scaler.inverse_transform(test_scaled[window_length:])[:,0]
print(just_methane_column.shape)

# +
anomalies = test_score_df[test_score_df.anomaly]

plt.plot(
    test[window_length:].index,
    just_methane_column,
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    anomalies.index,
    anomalies.methane_mixing_ratio_bias_corrected_mean,
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
# -









































# ### Multivariate AutoEncoder - Manually Windowed

def generate_datasets_for_training(data, window_size,scale=True):
    _l = len(data) 
    Xs = []
    Ys = []
    for i in range(0, (_l - window_size)):
        # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        Xs.append(data[i:i+window_size])
        Ys.append(data[i:i+window_size])
        
    Xs=np.array(Xs)
    Ys=np.array(Ys)    
    return  (Xs.shape[2], Xs, Ys)


# +
feature_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count','dist_away_2_mean', 'qa_val_mean']

train_input = train[feature_cols]
test_input = test[feature_cols]

# s_scaler = StandardScaler()
# s_scaler = s_scaler.fit(train_input)

mm_scaler = MinMaxScaler()
mm_scaler = mm_scaler.fit(train_input)

train_scaled = mm_scaler.transform(train_input)
test_scaled = mm_scaler.transform(test_input)



# +
num_feats_train, trainX, trainY = generate_datasets_for_training(train_scaled, 7)
num_feats_test, testX, testY = generate_datasets_for_training(test_scaled, 7)


assert num_feats_train == num_feats_test

# +
print(num_feats_train)
print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)

# +
epochs = 100
batch_size = 32
window_length = 7


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)


model = keras.Sequential()
model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', batch_input_shape=(None, window_length, num_feats_train), return_sequences=True, name='encoder_1'))
model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'))
model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3'))
model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))
model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_feats_train)))

model.compile(loss=tf.losses.MeanAbsoluteError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanSquaredError(),tf.losses.MeanAbsoluteError()]
             )

model.build()
print(model.summary())

# -

history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=100, batch_size=batch_size, shuffle=False, callbacks=[early_stop])


plt.title('mae Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()



# +
print(num_feats_train)
print(trainX.shape)
print(trainY.shape)

print(testX.shape)
print(testY.shape)
# -

# ### Select back just the methane, and maybe others but weight it for the loss calculation



# +
X_train_pred = model.predict(trainX)
print('predicted train shape:', X_train_pred.shape)       
print('original train shape:', trainX.shape)

train_mae_loss = np.mean(np.abs(X_train_pred, trainX), axis=1)
print('train_mae_loss shape: ', train_mae_loss.shape)
# -



# #### Question!!! 
#
# * I don't understand how these plots are plotted when there are 4 dimensions to the loss
#
# * spider graphs maybe to visualize
# * parallel line charts 
# * tsne projection see if dimensionality reduction may help 
#

# I'm not exactly sure how this is computed...??
sns.distplot(train_mae_loss, bins=50, kde=True)

X_test_pred = model.predict(testX)
print('predicted test shape:', X_test_pred.shape)
print('original test shape:', testX.shape)
test_mae_loss = np.mean(np.abs(X_test_pred, testX), axis=1)
print('test_mae_loss shape: ', test_mae_loss.shape)

# error for each example from our training set
sns.distplot(test_mae_loss, bins=50, kde=True)



# #### Question!!! 
#
# * Here when I look at the loss, it is loss with respect to each of the 4 features, can I take an average of the loss and is that a valid choice?

avg_test_mae_loss = test_mae_loss.mean(axis=1)

# +
ANOMALY_THRESHOLD = 0.8

test_score_df = pd.DataFrame(index=test[seq_size:].index)
test_score_df['avg_loss'] = avg_test_mae_loss
test_score_df['threshold'] = ANOMALY_THRESHOLD
test_score_df['anomaly'] = test_score_df.avg_loss > test_score_df.threshold
test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[seq_size:].methane_mixing_ratio_bias_corrected_mean
test_score_df['rn_lat_2'] = test[seq_size:].rn_lat_2
test_score_df['rn_lon_2'] = test[seq_size:].rn_lon_2
test_score_df['reading_count'] = test[seq_size:].reading_count
test_score_df['dist_away_2_mean'] = test[seq_size:].dist_away_2_mean 
test_score_df['qa_val_mean'] = test[seq_size:].qa_val_mean


plt.plot(test_score_df.index, test_score_df.avg_loss, label = 'avg_loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Avg Test Loss vs. Anomaly Loss Threshold")
plt.legend()

# -

test_score_df.head()

# #### Question!!! 
#
# * Is this valid for me to inverse transform my scalar operation, just extract the 

just_methane_column = mm_scaler.inverse_transform(test_scaled[seq_size:])[:,0]
print(just_methane_column.shape)

# +
anomalies = test_score_df[test_score_df.anomaly]

plt.plot(
    test[seq_size:].index,
    just_methane_column,
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    anomalies.index,
    anomalies.methane_mixing_ratio_bias_corrected_mean,
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
# -


