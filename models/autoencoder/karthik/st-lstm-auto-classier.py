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
resolution = 'rn'

#### #### #### #### ####

lat_str=''
lon_str=''
dist_away_str=''

if '1' in resolution:
    lat_str = 'rn_lat_1'
    lon_str = 'rn_lon_1'
    dist_away_str = 'dist_away_1_mean'
elif '2' in resolution:
    lat_str = 'rn_lat_2'
    lon_str = 'rn_lon_2'
    dist_away_str = 'dist_away_2_mean'
elif '5' in resolution:
    lat_str = 'rn_lat_5'
    lon_str = 'rn_lon_5'
    dist_away_str = 'dist_away_5_mean'

else:
    lat_str = 'rn_lat'
    lon_str = 'rn_lon'
    dist_away_str = 'dist_away_mean'


file_name=f'data_{frequency}_{resolution}.parquet.gzip' # Insert specific data variant file name here
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
# start_dt = df.index.min()
# end_dt = df.index.max()

# print(start_dt)
# print(end_dt)

# time_table = pd.DataFrame(pd.date_range(start=start_dt, end=end_dt), columns=['time_utc'])
# print(time_table.shape)
# time_table.head()

# +
# For - GPU
# time_table['key'] = 0
# df_g_lat_lon['key'] = 0
# all_time_lat_lon = time_table.merge(df_g_lat_lon, on='key', how='outer').drop('key', axis = 1)

# +
# all_time_lat_lon = time_table.merge(df_g_lat_lon, how='cross')
# all_time_lat_lon = all_time_lat_lon.sort_values(by=['time_utc', lat_str, lon_str], ascending=[True, False, True])

# print(all_time_lat_lon.shape)
# all_time_lat_lon.head()

# +
# df = df.reset_index()
# print(df.shape)
# df.head()

# +
# final_df = df.merge(all_time_lat_lon, on=['time_utc', lat_str, lon_str], how='outer')
# print(final_df.shape)
# final_df.head()

# +
# final_df.isnull().sum()

# +
# final_df.head()
# -

# ### Final Overall Shape:
#
#
# ### (None, 964, 7, 4)
#

# ## Train / Test Split

# +
final_df = df #df.set_index('time_utc')
train_date_threshold = '2021-01-01' 
validation_date_threshold = '2021-06-01'

train = final_df.loc[final_df.index < train_date_threshold]
validation = final_df.loc[(final_df.index >= train_date_threshold) & (final_df.index < validation_date_threshold)]
test = final_df.loc[final_df.index >= validation_date_threshold]

print(train.shape, validation.shape, test.shape)
# -

validation.head()

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


validation.head()

# -

print('Print # of time steps in train', train.index.nunique())
print()
print('Print # of time steps in validation', validation.index.nunique())
print()
print('Print # of time steps in test', test.index.nunique())
print()
print("frequency", frequency)

# +
# For - GPU Cleanup
# df_g_lat_lon = df_g_lat_lon.drop('key', axis = 1)
# time_table = df_g_lat_lon.drop('key', axis = 1)
# -

window_length = 30
# sorted_lat_lons = [tuple((pair[0], pair[1])) for pair in 
#                    df_g_lat_lon.values]
# mask_identifier = 0

# +
# def create_data_lstm_v1(data, window_length):
#     '''
#     Output shape ---> (None, 964, 7, 4)
#     '''
#     start= time.time()
    
#     #time
#     unique_time_steps = sorted(list(set(data.index.tolist())))
#     tot_time_length = len(unique_time_steps)
    
    
#     #geo
#     unique_lat_lons = [tuple((pair[0], pair[1])) for pair in df_g_lat_lon.values]
    
#     process_start= time.time()
#     print("Sort time taken: ", process_start - start)

#     print()
#     print('total timesteps', tot_time_length)

#     outer_ts_data= [] #shape = (tot_time_length, len(df_g_lat_lon), window_length, len(feature_cols)))
    
#     for outer_ts_index in range((tot_time_length - window_length)):
        
#         print("Outer TS ", unique_time_steps[outer_ts_index])
#         outer_ts_all_geo_data = [] #shape = (len(df_g_lat_lon), window_length, len(feature_cols)))
        
#         # df_g_lat_lon is already sorted to go top down of CA, left to right
#         for pair in sorted_lat_lons:
            
#             cur_lat, cur_lon = pair
#             df_cur_geo_day = final_df[(final_df[lat_str] == cur_lat) & (final_df[lon_str] == cur_lon)]
# #             print('cur lat/lon shape:', df_cur_geo_day.shape) #this should be number of (1038/frequency, number columns)
            
#             ## MIGHT NEED TO FILL NAN's
#             x_vals_inner_ts_windows = [] # shape = (window_length, len(feature_cols)
            
#             # For each time step inside the window of size `window_length`
#             for inner_ind, inner_ts in enumerate(unique_time_steps[outer_ts_index : outer_ts_index+window_length]):
#                 df_cur_inner_ts = df_cur_geo_day[df_cur_geo_day.index == inner_ts][feature_cols]
#                 x_vals_inner_ts_windows.append(df_cur_inner_ts.values[0]) #Had to add a zero index here!! 

#             outer_ts_all_geo_data.append(x_vals_inner_ts_windows)
        
# #         print("Adding to Outer TS Data", outer_ts_all_geo_data.shape)
#         outer_ts_data.append(outer_ts_all_geo_data)
    
#     end = time.time()
#     print(end-process_start)
#     return tf.constant(np.array(outer_ts_data)) , tf.constant(np.array(outer_ts_data))


# train_samp = train[:3000]
# val_samp = validation[:3000]
# test_samp = test[:3000]

# trainX, trainY = create_data_lstm_v1(train_samp, window_length)
# valX, valY = create_data_lstm_v1(val_samp, window_length)
# testX, testY = create_data_lstm_v1(test_samp, window_length)

# -

# ### BREAK 

print(final_df.shape)
final_df


# + jupyter={"outputs_hidden": true}
def create_data_lstm_v2(data, window_length):
    '''
    '''
    start= time.time()
    data_tracker = []
    
    #time
    unique_time_steps = sorted(list(set(data.index.tolist())))
    tot_time_length = len(unique_time_steps)
    
    
    #geo
    data_g_lat_lon = data.groupby([lat_str, lon_str]).size().reset_index()
    data_g_lat_lon = data_g_lat_lon[[lat_str, lon_str]].sort_values(by=[lat_str, lon_str], ascending=[False, True])
    unique_lat_lons = [tuple((pair[0], pair[1])) for pair in data_g_lat_lon.values]
    
    
    process_start= time.time()
    print("Sort time taken: ", process_start - start)

    print()
    print('total timesteps', tot_time_length)

    outer_ts_data= [] #shape = (tot_time_length, len(df_g_lat_lon), window_length, len(feature_cols)))
    
    for outer_ts_index in range((tot_time_length - window_length)):
        
        print("Outer TS ", unique_time_steps[outer_ts_index])        
        # df_g_lat_lon is already sorted to go top down of CA, left to right
        for pair in unique_lat_lons:
            
            cur_lat, cur_lon = pair
            df_cur_geo_day = data[(data[lat_str] == cur_lat) & (data[lon_str] == cur_lon)]
    
            if not df_cur_geo_day.empty:
                ## MIGHT NEED TO FILL NAN's
                x_vals_inner_ts_windows = [] # shape = (window_length, len(feature_cols)

                # For each time step inside the window of size `window_length`
                for inner_ind, inner_ts in enumerate(unique_time_steps[outer_ts_index : outer_ts_index+window_length]):
                    df_cur_inner_ts = df_cur_geo_day[df_cur_geo_day.index == inner_ts][feature_cols]
                    if not df_cur_inner_ts.empty:
                        x_vals = df_cur_inner_ts.values[0]
                    else:
                        x_vals = np.zeros(4)
                    x_vals_inner_ts_windows.append(x_vals)

                outer_ts_data.append(x_vals_inner_ts_windows)

                #For Tracking outcome labels
                cur_meta = (unique_time_steps[outer_ts_index], cur_lat, cur_lon)
                cur_meta_list = list((cur_meta,) * window_length)
                data_tracker.extend(cur_meta_list)
                
            else:
                print("Nothing in here 3", pair, "Outer TS", unique_time_steps[outer_ts_index])
    
    end = time.time()
    print(end-process_start)
    
    x_result = np.array(outer_ts_data)
    return x_result, x_result, data_tracker
#     return tf.constant(x_result), tf.constant(x_result)


# train_samp = train[:7500]
# val_samp = validation[:7500]
# test_samp = test[:7500]


trainX, trainY, trainTrack = create_data_lstm_v2(train, window_length)
valX, valY, valTrack= create_data_lstm_v2(validation, window_length)
testX, testY, testTrack = create_data_lstm_v2(test, window_length)

# -



# ### Check Data

print(trainX.shape)
print(valX.shape)
print(testX.shape)
print()
print(trainTrack[0], trainTrack[-1])
print(valTrack[0], valTrack[-1])
print(testTrack[0], testTrack[-1])


# +
def write_np(fpath, arr):
    # reshaping the array from 3D matrice to 2D matrice.
    arrReshaped = arr.reshape(arr.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt(fpath, arrReshaped)

def load_np(fpath, feature_count):
    # retrieving data from file.
    loadedArr = np.loadtxt(filename)
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // feature_count, feature_count)
    return loadedOriginal


# -

# ### Write Data locally if it took long to load

if False:
    local_path = 'windowed_data/'+ freq + "_" + resolution + "_" + str(window_length)
    print(local_path)

    if not os.path.exists(local_path): 
        os.mkdir(local_path)
    else:
        print('path exists')

    for d in ['train','val','test']:

        if d == 'train':

            write_np(local_path+'/'+d, trainX)
            textfile = open(f"{local_path}/trainTrack.txt", "w")
            for element in trainTrack:
                textfile.write(str(element) + "\n")
            textfile.close()        

        elif d == 'val':

            write_np(local_path+'/'+d, valX)

            textfile = open(f"{local_path}/valTrack.txt", "w")
            for element in valTrack:
                textfile.write(str(element) + "\n")
            textfile.close() 

        elif d == 'test':        
            write_np(local_path+'/'+d, testX)        
            textfile = open(f"{local_path}/testTrack.txt", "w")
            for element in testTrack:
                textfile.write(str(element) + "\n")
            textfile.close()


    textfile = open(f"{local_path}/feature_cols.txt", "w")
    for element in feature_cols:
        textfile.write(element + "\n")
    textfile.close()

    print("Finished!")

# # Things We Tried:
#
# * Masking with -999, masking keras layer, losses were really really high. Greater than those with 0 mask. 
#
# * masking with 0  --> : 104.7146 - mean_squared_error: 193878.4375 - mean_absolute_error: 105.1299 - val_loss: 89.5817 - val_mean_squared_error: 167090.7344 - val_mean_absolute_error: 90.1758
#
# * We only look at data that we have
#
# * We used Frequency 5D, resolution 0.5, and padded to make sure that the shape matched the window length
#     * Both these approaches have no regulairzaiton and only use 30% of data to train
#     * 143/143 [==============================] - 2s 16ms/step - loss: 0.0688 - mean_squared_error: 0.0191 - mean_absolute_error: 0.0687 - val_loss: 0.0595 - val_mean_squared_error: 0.0127 - val_mean_absolute_error: 0.0594
#     * with mask: 3s 18ms/step - loss: 0.0550 - mean_squared_error: 0.0116 - mean_absolute_error: 0.0550 - val_loss: 0.0485 - val_mean_squared_error: 0.0088 - val_mean_absolute_error: 0.0485
#     
#     * Able to complete the full model, but only able to achieve around 8% accuracy with the spatio temporal model on this data.

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

# ## Generic Modelling Config
#
# - Function to get model `run_id`

# +
dev_name = 'karthik'
assert dev_name != '', "Fill in your name"


from tensorflow.keras import backend as K

def getRunID(custom_name, cur_config):
    
    config_parts = [str(k) +"_" + str(v) for k, v in cur_config.items()]
    conf_str = "-".join(config_parts)
    model_name = f'ae_{custom_name}:{conf_str}_{frequency}_{resolution}_{dev_name}'
    print("Model Name: ", model_name)
    return model_name


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True)


def root_mean_squared_error_custom(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 


losses = [tf.losses.MeanAbsoluteError(), tf.losses.MeanSquaredError(), root_mean_squared_error_custom]

loss_to_use = losses[1]
track_metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.metrics.MeanSquaredError(), tf.losses.MeanAbsoluteError()]

input_shape = (window_length, len(feature_cols))
input_shape
# -

# ### Model 1

getRunID('100unitLSTM', config)



trainX.shape[0]/200

# +
# Define model config (Add all necessary model arguments and hyperparameters)
config = {
    'd_rate': 0.1,
    'optimizer': 'adam',
    'epochs': 100,
    'batch_size': 64
}

run_id = getRunID('100unitLSTM', config)

# Set up Tensorboard callback
log_dir = os.path.join(f"{ROOT_DIR}/logs/{dev_name}", run_id + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# +
model = keras.Sequential()
model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape))
model.add(keras.layers.LSTM(100, input_shape=input_shape, activation='relu', return_sequences=False, name='encoder_1'))
model.add(keras.layers.Dropout(rate=config['d_rate']))
model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))
model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True, name='decoder_1'))
model.add(keras.layers.Dropout(rate=config['d_rate']))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(feature_cols))))


model.compile(loss = loss_to_use,
              optimizer = config['optimizer'],
              metrics = track_metrics 
             )

# model.build()
model.summary()


# -

history = model.fit(x=trainX,
                    y=trainY,
                    validation_data=(valX, valY), 
                    epochs = config['epochs'],
                    batch_size = config['batch_size'],
                    shuffle=False, 
                    callbacks=[early_stopping, tensorboard_callback])


# ## Version 2

# +
# Define model config (Add all necessary model arguments and hyperparameters)
config = {
    'd_rate': 0.2,
    'optimizer': 'adam',
    'epochs': 50,
    'batch_size': 128,
}

run_id = getRunID('256_128_64_w_drate_no_init_unitLSTM', config)

# Set up Tensorboard callback
log_dir = os.path.join(f"{ROOT_DIR}/logs/{dev_name}", run_id + '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

print()
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# +
model = keras.Sequential()
model.add(tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape))
model.add(keras.layers.LSTM(256, activation='relu', return_sequences=True, name='encoder_1'))
model.add(keras.layers.Dropout(rate=config['d_rate']))
model.add(keras.layers.LSTM(128, activation='relu', return_sequences=True, name='encoder_2'))
model.add(keras.layers.Dropout(rate=config['d_rate']))
model.add(keras.layers.LSTM(64, activation='relu', return_sequences=False, name='encoder_3'))
model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))
model.add(keras.layers.LSTM(64, activation='relu', return_sequences=True, name='decoder_1'))
model.add(keras.layers.Dropout(rate=config['d_rate']))
model.add(keras.layers.LSTM(128, activation='relu', return_sequences=True, name='decoder_2'))
model.add(keras.layers.Dropout(rate=config['d_rate']))
model.add(keras.layers.LSTM(256, activation='relu', return_sequences=True, name='decoder_3'))
model.add(keras.layers.Dense(128, activation='relu', name='dense_1'))
model.add(keras.layers.Dense(64, activation='relu', name='dense_2'))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(len(feature_cols))))

model.compile(loss = loss_to_use,
              optimizer = config['optimizer'],
              metrics = track_metrics 
             )

model.summary()

# -

history = model.fit(x=trainX,
                    y=trainY,
                    validation_data=(valX, valY), 
                    epochs = config['epochs'],
                    batch_size = config['batch_size'],
                    shuffle=False, 
                    callbacks=[early_stopping, tensorboard_callback])










# ## Classifier Model

model.summary()

latent_output = model.get_layer('encoder_3').output
latent_output

len(trainTrack)

trainX.shape

# +
model_latent = keras.Model(inputs=model.inputs, outputs=latent_output)

# get the feature vector for the input sequence
next_trainX = model_latent.predict(trainX)
next_trainY = trainTrack[::window_length]

next_valX = model_latent.predict(valX)
next_valY = valTrack[::window_length]

next_testX = model_latent.predict(testX)
next_testY = testTrack[::window_length]


print(next_trainX.shape)
print(next_valX.shape)
print(next_testX.shape)

# -

next_trainX[123]

trainTrack[123]

# +
from collections import Counter

train_pairs = [str(x[1])+":"+str(x[2]) for x in trainTrack[::window_length]]
val_pairs = [str(x[1])+":"+str(x[2]) for x in valTrack[::window_length]]
test_pairs = [str(x[1])+":"+str(x[2]) for x in testTrack[::window_length]]

train_class_counts = Counter(train_pairs)
val_class_counts = Counter(val_pairs)
test_class_counts = Counter(test_pairs)


print("Mainly train needs to be larger than the others!")
print(len(train_class_counts))
print(len(val_class_counts))
print(len(test_class_counts))

num_classes = len(train_class_counts)

# -

# ### Create Geo ID Map

# +
geo_id_map = {}
#Save 0 for potentially unknown pairs....
for ind, pair_str in enumerate(train_pairs, 1):
    if pair_str not in geo_id_map:
        geo_id_map[pair_str] = ind

id_geo_map= dict((v,k) for k,v in geo_id_map.items())

print(len(geo_id_map))
print(len(id_geo_map))

# +
id_train_pairs = np.array([geo_id_map.get(pair_str, 0) for pair_str in train_pairs])
id_val_pairs = np.array([geo_id_map.get(pair_str, 0) for pair_str in val_pairs])
id_test_pairs = np.array([geo_id_map.get(pair_str, 0) for pair_str in test_pairs])

id_train_pairs = id_train_pairs.reshape((len(id_train_pairs), 1))
id_val_pairs = id_val_pairs.reshape((len(id_val_pairs), 1))
id_test_pairs = id_test_pairs.reshape((len(id_test_pairs), 1))

print(id_train_pairs.shape)
print(id_val_pairs.shape)
print(id_test_pairs.shape)
print()

t_next_trainY = keras.utils.to_categorical(id_train_pairs)
t_next_valY = keras.utils.to_categorical(id_val_pairs)
t_next_testY = keras.utils.to_categorical(id_test_pairs)

print(t_next_trainY.shape)
print(t_next_valY.shape)
print(t_next_testY.shape)


# -

# ### Input + Label for Classifier

print("X")
print(next_trainX.shape)
print(next_valX.shape)
print(next_testX.shape)
print()
print("Y")
print(t_next_trainY.shape)
print(t_next_valY.shape)
print(t_next_testY.shape)

# ### DNN Architecture

# +
dnn_epochs = 50
dnn_batch_size = 128

print(dnn_epochs)
print(dnn_batch_size)

# +
input_dim = next_trainX.shape[1]
num_classes = t_next_trainY.shape[1]

dnn_model = keras.Sequential()
dnn_model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
dnn_model.add(layers.Dropout(0.05))
dnn_model.add(layers.Dense(128, activation='relu'))
dnn_model.add(layers.Dropout(0.05))
dnn_model.add(layers.Dense(num_classes, activation='softmax'))

dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dnn_history = dnn_model.fit(next_trainX,
                            t_next_trainY,
                            epochs=dnn_epochs,
                            batch_size=dnn_batch_size)

# evaluate the keras model
_, accuracy = dnn_model.evaluate(next_trainX, t_next_trainY)
print('Accuracy: %.2f' % (accuracy*100))
# -

plt.title('Categorical CrossEntropy Loss')
plt.plot(dnn_history.history['loss'], label='train')
plt.legend()


plt.title('Accuracy')
plt.plot(dnn_history.history['accuracy'], label='train')
plt.legend()


dnn_model.evaluate(next_valX, t_next_valY)

dnn_model.evaluate(next_testX, t_next_testY)

print("X")
print(next_trainX.shape)
print(next_valX.shape)
print(next_testX.shape)
print()
print("Y")
print(t_next_trainY.shape)
print(t_next_valY.shape)
print(t_next_testY.shape)







# +
X_train_pred = model.predict(trainX2)[:,:,:1]
trainX_methane = trainX2[:,:,:1]

print('predicted train shape:', X_train_pred.shape)       
print('original train methane shape:', trainX_methane.shape)
train_mae_loss = np.mean(np.abs(X_train_pred, trainX_methane), axis=1)
print('train_mae_loss shape: ', train_mae_loss.shape)
sns.distplot(train_mae_loss, bins=50, kde=True)
# -

# # Predict on Validation

# +
X_val_pred = model.predict(valX2)[:,:,:1]
valX_methane = valX2[:,:,:1]

print('predicted val shape:', X_val_pred.shape)       
print('original val methane shape:', valX_methane.shape)
val_mae_loss = np.mean(np.abs(X_val_pred,  valX_methane), axis=1)
print('val_mae_loss shape: ', val_mae_loss.shape)
sns.distplot(val_mae_loss, bins=50, kde=True)

# -

# ### Predict on Test
#

# +
X_test_pred = model.predict(testX2)[:,:,:1]
testX_methane = testX2[:,:,:1]

print('predicted test shape:', X_test_pred.shape)
print('original test methane shape:', testX_methane.shape)
test_mae_loss = np.mean(np.abs(X_test_pred, testX_methane), axis=1)
print('test_mae_loss shape: ', test_mae_loss.shape)
# sns.distplot(test_mae_loss, bins=50, kde=True)


# + endofcell="--"
### Define Anomaly Threshold

ANOMALY_THRESHOLD = 0.82

# # +
val_score_df = pd.DataFrame(index=validation[window_length:].index)
val_score_df['loss'] = val_mae_loss
val_score_df['threshold'] = ANOMALY_THRESHOLD
val_score_df['anomaly'] = val_score_df.loss > val_score_df.threshold
val_score_df['methane_mixing_ratio_bias_corrected_mean'] = validation[window_length:].methane_mixing_ratio_bias_corrected_mean
val_score_df[lat_str] = validation[window_length:][lat_str]
val_score_df[lon_str] = validation[window_length:][lon_str]
val_score_df['reading_count'] = validation[window_length:].reading_count
val_score_df[dist_away_str] = validation[window_length:][dist_away_str]
val_score_df['qa_val_mean'] = validation[window_length:].qa_val_mean



plt.plot(val_score_df.index, val_score_df.loss, label = 'loss')
plt.plot(val_score_df.index, val_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Validation Loss vs. Anomaly Loss Threshold")
plt.legend()

# -
# --

# + endofcell="--"
# ### Plot Anomalies

# # +
val_anomalies = val_score_df[val_score_df.anomaly]
val_methane_column = mm_scaler.inverse_transform(val_scaled[window_length:])[:,0]


plt.title("Valiidation Methane")
plt.plot(
    validation[window_length:].index,
    val_methane_column,
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    val_anomalies.index,
    val_anomalies.methane_mixing_ratio_bias_corrected_mean,
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()

# # +
test_score_df = pd.DataFrame(index=test[window_length:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = ANOMALY_THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[window_length:].methane_mixing_ratio_bias_corrected_mean
test_score_df[lat_str] = test[window_length:][lat_str]
test_score_df[lon_str] = test[window_length:][lon_str]
test_score_df['reading_count'] = test[window_length:].reading_count
test_score_df[dist_away_str] = test[window_length:][dist_away_str]
test_score_df['qa_val_mean'] = test[window_length:].qa_val_mean



plt.plot(test_score_df.index, test_score_df.loss, label = 'loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Test Loss vs. Anomaly Loss Threshold")
plt.legend()

# -
# --

# +
# # +
test_anomalies = test_score_df[test_score_df.anomaly]
test_methane_column = mm_scaler.inverse_transform(test_scaled[window_length:])[:,0]

plt.title("Test Methane")
plt.plot(
    test[window_length:].index,
    test_methane_column,
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    test_anomalies.index,
    test_anomalies.methane_mixing_ratio_bias_corrected_mean,
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
# -













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
# -









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

# * VAE for image reconstruction, layeres that are de-convolution
# * Video VAEs also capture time 
#

# ### Things to Try:
# * Single or multiple LSTM layers




