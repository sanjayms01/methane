# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (Data Science)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/datascience-1.0
# ---

# # Pre-Trained LSTM Autoencoder Models
#
# This notebook is used to train LSTM Autoencoder Models and save to S3.    
# This notebook will train one model per California Climate Zone, total of 16 models.  
#
# Steps:  
# 1) Pull compiled dataset from S3  
# 2) Break dataset to train, validation, and test  
# 3) Standard Scaler and drop NA's. Store Standard Scaler model locally.  
# 4) Train models on train data and save locally.  
# 5) Push Standard Scaler model and each LSTM Autoencoder model to S3.  

# +
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import altair as alt
import datetime

import os
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
# import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
from descartes import PolygonPatch


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

#Altair puts a limit on plotting only 5000 rows from a pd.DataFrame. This line gets rid of that limit
alt.data_transformers.disable_max_rows()

# -


#Read in Data
s3_file_path = 's3://methane-capstone/data/dt=latest/data-zone-combined.parquet.gzip'
df = pd.read_parquet(s3_file_path)
df['time_utc'] = pd.to_datetime(df['time_utc'])
df


# +
#####################################
# Function to load data, given region
#####################################
def load_all_zone_data(describe=True):

    #Read in Data
    s3_file_path = 's3://methane-capstone/data/dt=latest/data-zone-combined.parquet.gzip'
    df = pd.read_parquet(s3_file_path)
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    
#     train_date_threshold = '2021-01-01'
#     validation_date_threshold = '2021-06-01'
    train_days_ago_threshold = 270            #270 days ago to 90 days ago = validation set, before 270 days ago is train data
    validation_days_ago_threshold = 90        #90 days ago to present would be test data

    #look at the date on last row of combined dataframe and set train/valid/test thresholds based on that
    from datetime import timedelta, date                                                                     
    train_date_threshold = str(df.time_utc.iloc[-1] - timedelta(train_days_ago_threshold))[:10]         
    validation_date_threshold  = str(df.time_utc.iloc[-1] - timedelta(validation_days_ago_threshold))[:10]   
    
    df = df.set_index('time_utc')
    train = df.loc[df.index < train_date_threshold]
    validation = df.loc[(df.index >= train_date_threshold) & (df.index < validation_date_threshold)]
    test = df.loc[df.index >= validation_date_threshold]

    #Print time range
    print("start_dt:", df.index.min(), "\nend_dt:", df.index.max(), "\nnumber_days:", df.index.max() - df.index.min(), "\n")
    print(df.shape, "\n")
    print(df.dtypes, "\n")
    print(train.shape, validation.shape, test.shape)
    
    return df, train, validation, test


####################################################################
# Function to generate trainx, trainy and return number of features 
# Window Function
####################################################################
def generate_datasets(data, window_size, describe=False):
    _l = len(data) 
    Xs = []
    Ys = []
    for i in range(0, (_l - window_size)):
        # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        Xs.append(data[i:i+window_size])
        Ys.append(data[i:i+window_size])
        
    Xs=np.array(Xs)
    Ys=np.array(Ys)    
    
    if describe:
        print(Xs.shape, Ys.shape)
    
    return (Xs.shape[2], Xs, Ys)


# -
# ### Data 
#     * Standardize
#     * Training
#     * Loss Calculation

# +
####################################################################
# Function to standard scaler the data
####################################################################
def standardize_data(train, validation, test, feature_cols, zone, describe=False, save=False):

    train_input = train[feature_cols]
    val_input = validation[feature_cols]
    test_input = test[feature_cols]

    scaler = StandardScaler()
    scaler = scaler.fit(train_input)

    train_scaled = scaler.transform(train_input)
    val_scaled = scaler.transform(val_input)
    test_scaled = scaler.transform(test_input)

    train_features = train_scaled
    val_features = val_scaled
    test_features = test_scaled

    if describe:
        print("train:", train_features.shape)
        print("val:", val_features.shape)
        print("test:", test_features.shape)
    
    if save:
        #save standardscaler model locally first (will push to s3 after)
        import datetime
        import pickle
        
        standard_scaler_name = f'ScalerModel_Zone{zone}'                              
        pickle.dump(scaler, open(f'/root/methane/models/autoencoder/models/pretrained/{standard_scaler_name}.pkl','wb'))
   
        
    return train_scaled, val_scaled, test_scaled, scaler


####################################################################
# Function to run multivariate neural network
####################################################################
def lstm_multi(trainX, trainY, valX, valY, window_length, num_features, batch_size, epochs, plot=False):
    
    #build model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=128, input_shape = (window_length, num_features)))              ############# UPDATE
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=window_length))
    model.add(keras.layers.LSTM(units=128, return_sequences=True))                                    ############# UPDATE
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_features)))

    #compile model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanSquaredError(),
                           tf.losses.MeanAbsoluteError(),
                           tf.metrics.RootMeanSquaredError()
                          ]
                 )

    #defined early stopping when training
    early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=1e-2,
                            patience=5,
                            verbose=0,
                            mode='auto',
                            baseline=None, 
                            restore_best_weights=True
                        )

    #show model summary
    #model.summary()

    #train and fit model
    history = model.fit(x=trainX,
                        y=trainY,
                        validation_data=(valX, valY),
                        epochs=epochs,
                        batch_size=batch_size, 
                        shuffle=False, 
                        callbacks=[early_stopping])

    if plot:
        plt.title('MAE Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()

    return model, history


def calculate_loss(feature_num, model, dataX):

    #Predict model and calculate MSE of the feature (0th feature = methane)    
    pred = model.predict(dataX)[:, :, feature_num]
    truth = dataX[:, :, feature_num]

    mse_loss = np.mean(np.square(pred -  truth), axis=1)     
    return mse_loss, pred 



# -

# ### Load Data

df, train, val, test = load_all_zone_data()

# ### Track Everything

# +
#select region and features
zones = [x for x in range(1,17)]

# Track predictions and losses for analysis across different features
feature_loss_tracker = {key: {'train':{}, 'val':{}, 'test':{}} for key in zones}

#Track all the data frames, raw and scaled
df_tracker = {}

#Track all the metrics from each model training cycle
model_metrics_tracker = {}

drop = False

#parameters:
feature_cols = ['methane_mixing_ratio_bias_corrected_mean',  'reading_count',
                 'air_pressure_at_mean_sea_level_mean',
                 'eastward_wind_at_100_metres_mean',
                 'northward_wind_at_100_metres_mean',
                 'air_temperature_at_2_metres_mean',
                 'surface_air_pressure_mean',
                 'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean',
                 'precipitation_amount_1hour_Accumulation_mean' ,
                 'dew_point_temperature_at_2_metres_mean']

feature_number_map = {}
for ind, feature in enumerate(feature_cols, 0):
    feature_number_map[feature] = ind

start=time.time()
    
for zone in zones:
    
    print("Zone #", zone)
    train_zone = train[train['BZone'] == zone]
    val_zone = val[val['BZone'] == zone]
    test_zone = test[test['BZone'] == zone] 
    
    if drop:
        #NEED TO DROP ROWS WITH NA VALUES :(
        train_zone=train_zone.dropna()
        val_zone=val_zone.dropna()
        test_zone=test_zone.dropna()
    
    else:
        train_zone=train_zone.interpolate(method='time')
        val_zone=val_zone.interpolate(method='time')
        test_zone=test_zone.interpolate(method='time')
        train_zone=train_zone.dropna()
        val_zone=val_zone.dropna()
        test_zone=test_zone.dropna()
        
    window_length = 7
    batch_size = 32
    num_features = len(feature_cols)
    epochs = 50

    print("Standard scaler'ing data")
    #standardize data
    train_scaled, val_scaled, test_scaled, scaler = standardize_data(train_zone, val_zone, test_zone, feature_cols, zone, save=True)
    
    
    #Track all data for use later on
    df_tracker[zone] = {'train_zone': train_zone,
                        'val_zone': val_zone,
                        'test_zone': test_zone,
                        'train_scaled': train_scaled,
                        'val_scaled': val_scaled,
                        'test_scaled': test_scaled,
                        'scaler': scaler
                       }

    print("Generating Datasets")
    #generate trainX and trainY
    num_feats_train, trainX, trainY = generate_datasets(train_scaled, window_length)
    num_feats_val, valX, valY = generate_datasets(val_scaled, window_length)
    num_feats_test, testX, testY = generate_datasets(test_scaled, window_length)
    
    assert num_feats_train == num_feats_test == num_feats_val
    
    print("training model")
    #Run LSTM Multivariate model and plot
    model, history =  lstm_multi(trainX, trainY, valX, valY, window_length, num_features, batch_size, epochs, plot=False)

    #Save model locally first, then push to S3 later
    import datetime
    cur_date = datetime.datetime.now().strftime("%Y%m%d")
    model_name = f'LSTMAE_Zone{zone}'                              
    print("saving model named:", model_name)
    model.save(f'/root/methane/models/autoencoder/models/pretrained/{model_name}.h5')   
    
    model_metrics_tracker[zone] = history.history

    for feature in feature_cols:

        #Predict MSE's:
        feature_num = feature_number_map[feature]
        print("Loss: ", feature, feature_num)
        
        train_mse_loss, X_train_pred = calculate_loss(feature_num, model, trainX)
        val_mse_loss, X_val_pred = calculate_loss(feature_num, model, valX)
        test_mse_loss, X_test_pred = calculate_loss(feature_num, model, testX)
        
        feature_loss_tracker[zone]['train'].update({feature: {'train_mse_loss': train_mse_loss, 'X_train_pred':X_train_pred }})
        feature_loss_tracker[zone]['val'].update({feature: {'val_mse_loss': val_mse_loss, 'X_val_pred': X_val_pred }})
        feature_loss_tracker[zone]['test'].update({feature: {'test_mse_loss': test_mse_loss, 'X_test_pred':X_test_pred }})

    print()
    print()
    print()

end=time.time()
print("TIME: {time:.2f} secs".format(time=(end-start)))
# +
# Build Final Dataframes For Visuals

def get_anomaly_threshold(mse_loss):
    upper,lower = np.percentile(mse_loss,[75,25])
    ANOMALY_THRESHOLD = 5*(upper-lower)
    return ANOMALY_THRESHOLD

# plot MSE for Train and Validation
final_dataframes = {key: {'train':None, 'val': None, 'test': None} for key in zones}
anomaly_thresholds = {key: {feature:None} for key in zones}

for zone in zones:
    for split in ['train', 'val', 'test']:
            
        cur_zone_df = df_tracker[zone][f'{split}_zone']
        scored_df = pd.DataFrame(index=cur_zone_df[window_length:].index)
    
        for feature in feature_cols:
            
            ### WE MIGHT HAVE TO FIGURE OUT THE COLOR BUCKETS HERE ###
            
            train_mse_loss = feature_loss_tracker[zone]['train'][feature][f'train_mse_loss']
            mse_loss = feature_loss_tracker[zone][split][feature][f'{split}_mse_loss']
            
            anom_thresh = get_anomaly_threshold(train_mse_loss)
            anomaly_thresholds[zone][feature] = anom_thresh

            
            scored_df[feature] = cur_zone_df[window_length:][feature]
            scored_df[f'{feature}_loss'] = mse_loss
            scored_df[f'{feature}_threshold'] = anom_thresh
            scored_df[f'{feature}_anomaly'] = scored_df[f'{feature}_loss'] > scored_df[f'{feature}_threshold']

        final_dataframes[zone][split] = scored_df

# +
# Save Dictionaries to Local

#### Dictionaries ####
# feature_loss_tracker
# df_tracker
# model_metrics_tracker

import boto3
import pickle

#save to local
with open(f'/root/methane/models/autoencoder/models/zone_artifacts/feature_loss_tracker.pickle', 'wb') as handle:
    pickle.dump(feature_loss_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'/root/methane/models/autoencoder/models/zone_artifacts/df_tracker.pickle', 'wb') as handle:
    pickle.dump(df_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'/root/methane/models/autoencoder/models/zone_artifacts/model_metrics_tracker.pickle', 'wb') as handle:
    pickle.dump(model_metrics_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'/root/methane/models/autoencoder/models/zone_artifacts/final_dataframes.pickle', 'wb') as handle:
    pickle.dump(final_dataframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'/root/methane/models/autoencoder/models/pretrained/pretrained_anomaly_thresholds.pickle', 'wb') as handle:
    pickle.dump(anomaly_thresholds, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
# -

# # Push Zone Artifacts and Pretrained Models to S3

# +
import subprocess

#save pretrained models to latest folder
subprocess.check_output(['aws','s3','cp', '--recursive', '/root/methane/models/autoencoder/models' , 's3://methane-capstone/models/autoencoder/dt=latest/'])


#save pretrained models to archive folder
from datetime import timedelta, date
localtime = time.localtime(time.time())
date=str(localtime.tm_year)+str(localtime.tm_mon)+str(localtime.tm_mday) #define date (for naming backup)
s3_location = f's3://methane-capstone/models/autoencoder/dt=archive/dt={date}/'
subprocess.check_output(['aws','s3','cp', '--recursive', '/root/methane/models/autoencoder/models' , s3_location])
# -


