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



# +
#####################################
# Function to load data, given region
#####################################
def load_all_zone_data(describe=True):

    #Read in Data
    s3_file_path = 's3://methane-capstone/data/combined-raw-data/data-zone-combined.parquet.gzip'
    df = pd.read_parquet(s3_file_path)
    df['time_utc'] = pd.to_datetime(df['time_utc'])

    df = df.set_index('time_utc')

    train_date_threshold = '2021-01-01'
    validation_date_threshold = '2021-06-01'

    train = df.loc[df.index < train_date_threshold]
    validation = df.loc[(df.index >= train_date_threshold) & (df.index < validation_date_threshold)]
    test = df.loc[df.index >= validation_date_threshold]

    if describe:
        #Print time range
        print("start_dt:", df.index.min(), "\nend_dt:", df.index.max(), "\nnumber_days:", df.index.max() - df.index.min(), "\n")

        print(df.shape, "\n")
        print(df.dtypes, "\n")        
        print(train.shape, validation.shape, test.shape)
        df.head()
    
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
    
    return  (Xs.shape[2], Xs, Ys)
# -

# ### Load Data



df, train, val, test = load_all_zone_data()



# +
####################################################################
# Function to standard scaler the data
####################################################################
def standardize_data(train, validation, test, feature_cols, describe=False):

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

    return train_scaled, val_scaled, test_scaled, scaler


####################################################################
# Function to run multivariate neural network
####################################################################
def lstm_multi(trainX, trainY, valX, valY, window_length, num_features, batch_size, epochs, plot=False):
    
    #build model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=64, input_shape = (window_length, num_features)))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=window_length))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_features)))

    #compile model
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanSquaredError(),tf.losses.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
                 )

    #defined early stopping when training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True)

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
    pred = model.predict(dataX)[:,:,feature_num]    
    truth = dataX[:,:,feature_num]

    mse_loss = np.mean(np.square(pred -  truth), axis=1)     
    return mse_loss, pred 

####################################################################
# Functions to plot
####################################################################

def plotting_distplot(train_mse_loss, val_mse_loss, feature, describe=False):
    
    #MSE LOSS
#     print("Train MSE loss distribution")
    print("Plots are for feature {}".format(feature))
    sns.distplot(train_mse_loss, bins=50, kde=True)
#     print("Validation MSE loss distribution")
    sns.distplot(val_mse_loss, bins=50, kde=True)
    plt.legend(labels=["TrainMSE","ValMSE"])
    
    if describe:
        print('predicted train shape:', X_train_pred.shape)       
        print('original train methane shape:', trainX_methane.shape)
        print('train_mse_loss shape: ', train_mse_loss.shape)
        print('predicted val shape:', X_val_pred.shape)       
        print('original val methane shape:', valX_methane.shape)
        print('val_mse_loss shape: ', val_mse_loss.shape)


def model_analysis_plots(region, train_mse_loss, ANOMALY_THRESHOLD, val_score_df,
                         validation, val_scaled, val_anomalies, mm_scaler, feature, save=False):
    
    #combined plots
    fig, axs = plt.subplots(3, 1, figsize=(10,15))  #specify how many sub - plots in figure

    plt. subplots_adjust(hspace=0.5)   #spacing between each subplot
    titles_font_size = 12
    ticks_font_size = 8
    fig.suptitle("LSTM AE Multivariate Model at Region {}, Feature {}".format(region, feature))  #title for entire subplot

    #Training Losses
    plt.setp(axs[0].get_xticklabels(), fontsize=ticks_font_size, rotation=25)#, horizontalalignment="left")
    plt.setp(axs[0].get_yticklabels(), fontsize=ticks_font_size)#,  horizontalalignment="left")
    axs[0].hist(train_mse_loss, bins=50, label='MSE Frequency')
    axs[0].axvline(ANOMALY_THRESHOLD, color = 'orange', label= 'threshold')
    axs[0].set_title("Histogram of Training Losses", fontsize=titles_font_size)
    axs[0].set_xlabel("Mean Squared Error", fontsize = titles_font_size) 
    axs[0].set_ylabel("Frequency", fontsize = titles_font_size) 
    axs[0].legend(fontsize=titles_font_size)


    #Validation Errors
    plt.setp(axs[1].get_xticklabels(), fontsize=ticks_font_size, rotation=25)#, horizontalalignment="left")
    plt.setp(axs[1].get_yticklabels(), fontsize=ticks_font_size)#,  horizontalalignment="left")
    axs[1].plot(val_score_df.index, val_score_df.loss, label = 'loss')
    axs[1].plot(val_score_df.index, val_score_df.threshold, label = 'threshold')
    axs[1].set_title("Validation Loss vs. Anomaly Loss Threshold", fontsize=titles_font_size)
    axs[1].legend(fontsize=10)
    axs[1].set_ylabel("Frequency", fontsize = titles_font_size) 

    #Validation Methane
    feature_num = feature_number[feature]
    plt.setp(axs[2].get_xticklabels(), fontsize=ticks_font_size, rotation=25)#, horizontalalignment="left")
    plt.setp(axs[2].get_yticklabels(), fontsize=ticks_font_size)#,  horizontalalignment="left")
    axs[2].set_title("Validation Methane", fontsize=titles_font_size)
    axs[2].plot(validation[window_length:].index,  mm_scaler.inverse_transform(val_scaled[window_length:])[:,feature_num],  label = feature )
    axs[2].scatter(
        val_anomalies.index,
        val_anomalies[feature],
        color = sns.color_palette()[3],
        s=10,
        label='anomaly')
    axs[2].legend(fontsize=titles_font_size)

    if save:
        feature_num = feature_number[feature]
        fig.savefig("./figures/multivariate/lstmae_multivariate_region{}_feature{}".format(str(region), feature))
    
#     fig.clear(True)
    
####################################################################
# Functions for Anomaly Detection
####################################################################

def anomaly(train_mse_loss, other_mse_loss, train, other_data):

    upper,lower = np.percentile(train_mse_loss,[75,25])
    ANOMALY_THRESHOLD = 5*(upper-lower)

    other_score_df = pd.DataFrame(index=other_data[window_length:].index)
    other_score_df['loss'] = other_mse_loss
    other_score_df['threshold'] = ANOMALY_THRESHOLD
    other_score_df['anomaly'] = other_score_df.loss > other_score_df.threshold
    
    ## THIS IS WHERE WE CAN DETERMINE THE BUCKETING ## 
    
    for feature in feature_cols:
        other_score_df[feature] = other_data[window_length:][feature]
        
    other_anomalies = other_score_df[other_score_df.anomaly]
    
    return other_score_df, other_anomalies, ANOMALY_THRESHOLD


# -

# ### Track Everything

# + jupyter={"outputs_hidden": true}
#select region and features
zones = [x for x in range(1,17)]

# Track predictions and losses for analysis across different features
feature_loss_tracker = {key: {'train':{}, 'val':{}, 'test':{}} for key in zones}

#Track all the data frames, raw and scaled
df_tracker = {key: {} for key in zones}


#Track all the metrics from each model training cycle
model_metrics_tracker = {key: {} for key in zones}

drop = False

#parameters:
feature_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count', 'qa_val_mean','qa_val_mode',
               'air_pressure_at_mean_sea_level_mean',
               'air_temperature_at_2_metres_mean', 'air_temperature_at_2_metres_1hour_Maximum_mean', 'air_temperature_at_2_metres_1hour_Minimum_mean',
               'dew_point_temperature_at_2_metres_mean',
               'eastward_wind_at_100_metres_mean', 'eastward_wind_at_10_metres_mean',
               'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean',
               'lwe_thickness_of_surface_snow_amount_mean',
               'northward_wind_at_100_metres_mean', 'northward_wind_at_10_metres_mean',
               'precipitation_amount_1hour_Accumulation_mean', 'snow_density_mean',
               'surface_air_pressure_mean']

feature_number = dict()
for ind, feature in enumerate(feature_cols, 0):
    feature_number[feature] = ind

start=time.time()
    
for zone in zones:
    
    print("Zone #", zone)
    train_zone, val_zone, test_zone = train[train['BZone'] == zone],  val[val['BZone'] == zone],  test[test['BZone'] == zone]
    
    if drop:
        #NEED TO DROP ROWS WITH NA VALUES :(
        train_zone=train_zone.dropna()
        val_zone=val_zone.dropna()
        test_zone=test_zone.dropna()
    
    else:

        train_zone=train_zone.interpolate(method='time')
        val_zone=train_zone.interpolate(method='time')
        test_zone=test_zone.interpolate(method='time')

    
    window_length = 7
    batch_size = 32
    num_features = len(feature_cols)
    epochs = 50

    print("Standard scaler'ing data")
    #standardize data
    train_scaled, val_scaled, test_scaled, scaler = standardize_data(train_zone, val_zone, test_zone, feature_cols)
    
    
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

    model_metrics_tracker[zone] = history.history

    
    for feature in feature_cols:

        #Predict MSE's:
        feature_num = feature_number[feature]
        print("Loss: ", feature, feature_num)
        
        train_mse_loss, X_train_pred = calculate_loss(feature_num, model, trainX)
        val_mse_loss, X_val_pred = calculate_loss(feature_num, model, valX)
        test_mse_loss, X_test_pred = calculate_loss(feature_num, model, testX)
        
        feature_loss_tracker[zone]['train'].update({feature: {'train_mse_loss': train_mse_loss, 'X_train_pred':X_train_pred }})
        feature_loss_tracker[zone]['val'].update({feature: {'val_mse_loss': val_mse_loss, 'X_val_pred':X_val_pred }})
        feature_loss_tracker[zone]['test'].update({feature: {'test_mse_loss': test_mse_loss, 'X_test_pred':X_test_pred }})

    print()
    print()
    print()

end=time.time()
print("TIME: {time:.2f} secs".format(time=(end-start)))
# -
# ### Dictionaries
#
# **Track predictions and losses for analysis across different features**
# `feature_loss_tracker`
#
# **Track all the data frames, raw and scaled**
# `df_tracker`
#
#
# **Track all the metrics from each model training cycle**
# `model_metrics_tracker`
#

# +
import boto3
import pickle


# #Connect to S3 default profile
s3 = boto3.client('s3')

# serializedMyData = pickle.dumps(myDictionary)

# s3.put_object(Bucket='mytestbucket',Key='myDictionary').put(Body=serializedMyData)
# -

# ### Write out to Pickle Files

# +
# feature_loss_tracker
# df_tracker
# model_metrics_tracker


# with open('zone_artifacts_20211101/feature_loss_tracker.pickle', 'wb') as handle:
#     pickle.dump(feature_loss_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('zone_artifacts_20211101/df_tracker.pickle', 'wb') as handle:
#     pickle.dump(df_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('zone_artifacts_20211101/model_metrics_tracker.pickle', 'wb') as handle:
#     pickle.dump(model_metrics_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)



## LOAD PICKLE FILE
with open('zone_artifacts_20211101/df_tracker.pickle', 'rb') as handle:
    b = pickle.load(handle)

# print a == b
# -

# ### Put on S3

# import subprocess
# subprocess.check_output(['aws','s3','cp', '--recursive', 'zone_artifacts_20211101' , 's3://methane-capstone/models/autoencoder/zone_model_artifacts/'])




# ### Use Dictionaries for Analysis

# +
# plot MSE for Train and Validation

zone = 5
feat = 'methane_mixing_ratio_bias_corrected_mean'
train_mse_loss = feature_loss_tracker[zone]['train']['methane_mixing_ratio_bias_corrected_mean']['train_mse_loss']
val_mse_loss = feature_loss_tracker[zone]['val']['methane_mixing_ratio_bias_corrected_mean']['val_mse_loss']
train_zone = df_tracker[zone]['train_zone']
val_zone = df_tracker[zone]['val_zone']
val_scaled = df_tracker[zone]['val_scaled']
scaler = df_tracker[zone]['scaler']


plotting_distplot(train_mse_loss,
                  val_mse_loss,
                  feat
                 )
# -



# ### Look at Anomalies

val_score_df, val_anomalies, ANOMALY_THRESHOLD = anomaly(train_mse_loss, val_mse_loss, train_zone, val_zone)

# ### Plot Analysis Plots

model_analysis_plots(zone, 
                     train_mse_loss,
                     ANOMALY_THRESHOLD,
                     val_score_df, 
                     val_zone,
                     val_scaled, 
                     val_anomalies,
                     scaler,
                     feat,
                     save=False)

