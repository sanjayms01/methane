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

# # ML Pipeline Template

# This notebook is to be referenced as a template for the ML pipelines we build for the autoencoder framework.

# ## Developer Name: --Karthik--
# ## Date: --10/18/2021--

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
df = df.set_index('time_utc')
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



# ### Treat Lat/Lon as a categorical

df[[lat_str, lon_str]].nunique()

df_g_lat_lon = df.groupby([lat_str, lon_str]).size().reset_index()
df_g_lat_lon = df_g_lat_lon[[lat_str, lon_str]].sort_values(by=[lat_str, lon_str], ascending=[False, True])
print(df_g_lat_lon.shape)
df_g_lat_lon.head()

# +
start_dt = df.index.min()
end_dt = df.index.max()

print(start_dt)
print(end_dt)

time_table = pd.DataFrame(pd.date_range(start=start_dt, end=end_dt), columns=['time_utc'])
print(time_table.shape)
time_table.head()
# -

all_time_lat_lon = time_table.merge(df_g_lat_lon, how='cross')
print(all_time_lat_lon.shape)
all_time_lat_lon.head()


df = df.reset_index()
print(df.shape)
df.head()

# +
final_df = df.merge(all_time_lat_lon, on=['time_utc', lat_str, lon_str], how='outer')



print(final_df.shape)
final_df.head()
# -

final_df.isnull().sum()

final_df.head()

final_df[[lat_str, lon_str]].sort_values(by=[lat_str, lon_str], ascending=[False, True])



boom.shape

len(set(df.index))

(1038, 964, 7, 4)



# ### Todos:
#
# * fill in time rows
# * fill in all combination of lat/lon rows
# * if we get at runtime a poitn we don't have in our classes, find the closest class and treat it as that





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
feature_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count', dist_away_str, 'qa_val_mean']

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

for ind, row in df_g_lat_lon.iterrows():
    print(row[lat_str], row[lon_str])
    break



# +
def create_data_lstm(data, window_length):

    '''
    Output shape ---> (None, 964, 7, 4)
    '''
    
    
    start= time.time()    
    unique_time_steps = sorted(list(set(data.index.tolist())))
    tot_time_length = len(unique_time_steps) 
    windowed_time_range = tot_time_length - window_size
    print()
    print('total timesteps', tot_time_length)

    x_values = []
    y_values = []
    
    
    for i in range(windowed_time_range):
        
        cur_time_step = []
    
        for ind, row in df_g_lat_lon.iterrows():

            cur_lat, cur_lon = row[lat_str], row[lon_str]

            cur_geo_day_df = final_df[ (final_df[lat_str] == cur_lat) & (final_df[lon_str] == cur_lon)]
            print('cur lat/lon shape:', cur_geo_day_df.shape) #this should be data.shape


            x_vals_ts_windows = []

            for i in range(windowed_time_range):
                #print(i)
                x_vals_ts_windows.append(cur_geo_day_df.iloc[i:(i+window_length)].values)
            
            
            
            
            

#             #Readings
#             cur_day_data = tf.cast(tf.constant(cur_day_df.iloc[:, :].values), tf.float64)

# #             print("cur_day_data.shape", cur_day_data.shape)

#             #Paddings
#             pad_row, pad_col  = cur_day_data.shape[0], cur_day_data.shape[1]
#             #print('pad_row', pad_row)
#             #print('pad_col', pad_col)
            
#             pad = tf.zeros(
#                 [rounded_max_reading_per_day-pad_row, pad_col], dtype=tf.float64
#             )

# #             print("pad.shape", pad.shape)
# #             print()

#             cur_day_x = tf.concat([cur_day_data, pad], 0)            
#             x_values_cur_ts_window.append(cur_day_x)

#         x_values.append(x_values_cur_ts_window)

#     end= time.time()
#     print(end-start, '\n')

#     print(type(x_values))
#     print(len(x_values))
    
#     #Input + Output are the same!
#     x_result = tf.expand_dims(x_values, 0)
#     y_result = tf.expand_dims(x_values, 0)

#     print('x_result.shape', x_result.shape)
#     print('y_result.shape', y_result.shape)

#     return x_result, y_result



create_sequences_advanced(train, window_length)



# trainX, trainY = create_sequences_advanced(train, window_length)
# valX, valY = create_sequences_advanced(validation, window_length)
# testX, testY = create_sequences_advanced(test, window_length)



# -

def create_data_dnn(data, window_length):
    return



print('(number of training examples, time sequence window, number of rows per day, number of features)')
print()
print(trainX.shape)



# * Check that predicting 0s on the padding doesn't mess up the loss calcualtion
#     * look at masking methods here, only use the data thats there 



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

trainX.shape[1:]



# * VAE for image reconstruction, layeres that are de-convolution
# * Video VAEs also capture time 
#





# +
#Input shape
inputs = keras.Input(shape=trainX.shape[1:])

### ### Neural Network Begins ### ### 
dense_1 = layers.Dense(128, activation='relu')
x = dense_1(inputs)

dense_2 = layers.Dense(128, activation='relu')
x = dense_2(x)

avg_pool = layers.TimeDistributed(layers.GlobalAveragePooling1D()) #Here we can do flatten as well,
# max_pool = layers.MaxPooling1D(pool_size=2)
#Weight average of max vs. avg. ---> GEM pool

output_avg = avg_pool(x)
# output_max = max_pool(x)

# x = layers.Concatenate(output_avg, output_max) #to join the maxpool and average pool 
# layer_3 = layers.Flatten()

# Create the layers of the model
lstm_enc_1 = keras.layers.LSTM(units=64,)(output_avg)
# lstm_enc_2 = keras.layers.LSTM(units=128)(lstm_enc_1)

drop_1 = keras.layers.Dropout(rate=0.2)(lstm_enc_1)
repeat = keras.layers.RepeatVector(n=trainX.shape[1])(drop_1)
lstm_dec_2 = keras.layers.LSTM(units=64, return_sequences=True)(repeat)
drop_2 = keras.layers.Dropout(rate=0.2)(lstm_dec_2)
dense_3 = keras.layers.Dense(units=32)(drop_2)
dense_4 = keras.layers.Dense(units=1)(dense_3)
outputs = dense_4

model = keras.Model(inputs=inputs, outputs=outputs, name="methane_spatio_temporal")
model.summary()
# -

# * Use auto-encoder pre-training to learn back the input vector
# * with distance attention 

# Compile the model
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

model.fit(
    trainX, trainY,
    epochs = 5,
    validation_split = 0.2,
    shuffle = False,
)










# ## Define Model Config

# Define model config (Add all necessary model arguments and hyperparameters)
config = {
    'units': 64,
    'd_rate': 0.2,
    'loss': 'mae',
    'optimizer': 'adam',
    'epochs': 2,
    'batch_size': 32,
    'val_split': 0.1
}

# ## Define Model Name

# +
dev_name = '' # Put your name here
name_list = file_name.split('.')
freq_res = name_list[0].split('_')
freq = freq_res[1]

if (len(freq_res) == 4):
    resolution = freq_res[2] + freq_res[3]
else:
    resolution = freq_res[2]
    
model_name = f'autoencoder_{dev_name}_{freq}_{resolution}'
print("Model Name: ", model_name)


# -

# ## Create Model

def create_model(config):
    # Initialize the model
    model = keras.Sequential()
    
    # Get model config params
    units = config['units']
    rate = config['d_rate']

    # Create the layers of the model
    model.add(keras.layers.LSTM(units=units, input_shape = (trainX.shape[1], trainX.shape[2])))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.RepeatVector(n=trainX.shape[1]))
    model.add(keras.layers.LSTM(units=units, return_sequences=True))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense( units=trainX.shape[2])))
    
    return model


# Create the model and display its summary
model = create_model(config)
model.summary()


# ## Train Model

def train_model(config):
    # Create the model
    model = create_model(config)
    
    # Get model config values
    loss = config['loss']
    optimizer = config['optimizer']
    epochs = config['epochs']
    batch_size = config['batch_size']
    val_split = config['val_split']
    
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    # Set up Tensorboard callback
    log_dir = os.path.join(f"{ROOT_DIR}/logs/{dev_name}", model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(
        trainX, trainY,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split,
        shuffle = False,
        callbacks=[tensorboard_callback]
    )
    
    return model


# +
model = train_model(config)

# View training and validation loss
training_loss = model.history.history['loss']
print("Training Loss: ", training_loss)

validation_loss = model.history.history['val_loss']
print("Validation Loss: ", validation_loss)
# -

# ## Launch TensorBoard

# Please open up a system terminal and run the following commands:
# > cd ~/methane/models
#
# > pip install tensorboard
#
# > tensorboard --logdir logs/your-name. Replace `your-name` with developer name.
#
# > Visit the URL: https://d-kdgirgbbdmbt.studio.us-west-2.sagemaker.aws/jupyter/default/proxy/insert-port-number-here/. Replace `insert-port-number-here` with the port number shown in terminal output.
#
# > Ctrl C on terminal to exit

# ## Hyperparameter Tuning

# +
# Add more values to arrays to use different hyperparameters and add other parameter variables as necessary
units = []
d_rate = []
loss = []
optimizer = []
epochs = []
batch_size = []
val_split = []

# Create combinations of all hyperparameters
hyperparams = list(product(units, d_rate, loss, optimizer, epochs, batch_size, val_split))
print(hyperparams)
print(len(hyperparams))

# +
# Write hyperparameters to a dataframe (Add key-value pair for every hyperparameter you want to include in DF)
hp = {
    'units': [i[0] for i in hyperparams],
    'd_rate': [i[1] for i in hyperparams],
    'loss': [i[2] for i in hyperparams],
    'optimizer': [i[3] for i in hyperparams],
    'epochs': [i[4] for i in hyperparams],
    'batch_size': [i[5] for i in hyperparams],
    'val_split': [i[6] for i in hyperparams]
}

hp_df = pd.DataFrame(hp, columns = list(hp.keys()))
hp_df
# -

# #### Get Config For Each Model

configs = hp_df.to_dict(orient='records')
configs

# #### Create And Train Models Using Defined Hyperparameters

# +
trained_models = []
count = 0

for config in configs:
    count += 1
    print(f'Model {count}: {list(config.items())}')
    model = train_model(config)
    trained_models.append((config, model, model.history))
    print()
# -

# #### Launch TensorBoard
#
# Follow the steps described above to re-launch tensorboard to view model performance.

# +
# Add training results into new columns in hyperparameter dataframe
training_loss = []
val_loss = []
training_mae = []
val_mae = []

# Loop through the model histories
for hist in trained_models:
    num_epochs = hist[0]['epochs']
    training_loss.append(hist[2].history['loss'][num_epochs - 1])
    training_mae.append(hist[2].history['mae'][num_epochs - 1])
    val_loss.append(hist[2].history['val_loss'][num_epochs - 1])
    val_mae.append(hist[2].history['mae'][num_epochs - 1])
    
# Add new loss columns to hyperparameters dataframe (results table)
hp_df['training_loss'] = training_loss
hp_df['validation_loss'] = val_loss
hp_df['training_mae'] = training_mae
hp_df['validation_mae'] = val_mae

hp_df
# -

# Write hyperparameters dataframe to csv file
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
hp_df.to_csv('/root/methane/models/autoencoder/' + dev_name + "/" + model_name + cur_date + '.csv', index = False, header = True)

# ## Save Best Model

# Find model with lowest validation mae to save the best model
best_model_index = hp_df[hp_df.validation_mae == hp_df.validation_mae.min()].index
best_model = trained_models[best_model_index[0]][1]
best_model_name = f'{model_name}_{cur_date}'
best_model.save(f'/root/methane/models/autoencoder/{dev_name}/{best_model_name}.h5')

# ## Upload Best Model Artifact To S3

client = boto3.client('s3')
client.upload_file(Filename=f'/root/methane/models/autoencoder/{dev_name}/{best_model_name}.h5', Bucket=bucket, Key=f'models/autoencoder/{best_model_name}.h5')

# ## Plot Best Model

# +
# plot_model(best_model, to_file = f'./{model_name}/images/model_{}_{}.png', show_shapes=True, show_layer_names=True)
# -

# ## Evaluate Best Model On Test Data

best_model = model

best_model.evaluate(testX, testY)

# ## Predict On Test Data

testY.shape

testX_pred.shape

testX_pred = best_model.predict(testX)
test_mae_loss = np.mean(np.abs(testX_pred, testX), axis=1)

# ## Anomaly Detection

# +
"""
Selecting the threshold is key!
    - We can examine the training losses and let that guide our choice for the threshold
    - If we lower the threshold value, more anomalies will be detected
    - We can reason about this using descriptive statistics we have learned from the data
"""

anom_thresh = 1.5

test_score_df = pd.DataFrame(index=test[seq_size:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = anom_thresh
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[seq_size:].methane_mixing_ratio_mean


# Plot the Test Loss against Anomaly Loss Threshold
plt.plot(test_score_df.index, test_score_df.loss, label = 'loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Test Loss vs. Anomaly Loss Threshold")
plt.legend()

anomalies = test_score_df[test_score_df.anomaly]
print("Anomalies DF Shape: ", anomalies.shape)
anomalies

# +
plt.plot(
    test[seq_size:].index,
    scaler.inverse_transform(test[seq_size:].mmrbc_i),
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.mmrbc_i),
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
