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
# -

#Altair puts a limit on plotting only 5000 rows from a pd.DataFrame. This line gets rid of that limit
alt.data_transformers.disable_max_rows()


# ### Functions

# +
#####################################
# Function to load data, given region
#####################################
def load_data(region, describe=False):

    #Read in Data
    s3_file_path = 's3://methane-capstone/data/data-variants-zone/data_{}.parquet.gzip'.format(region)
    df = pd.read_parquet(s3_file_path)
    df['time_utc'] = pd.to_datetime(df['time_utc'])

    print(df.shape, "\n")
    print(df.dtypes, "\n")

    #Print time range
    print("start_dt:", df['time_utc'].min(), "\nend_dt:", df['time_utc'].max(), "\nnumber_days:", df['time_utc'].max() - df['time_utc'].min(), "\n")
    df = df.set_index('time_utc')

    train_date_threshold = '2021-01-01'
    validation_date_threshold = '2021-06-01'

    train = df.loc[df.index < train_date_threshold]
    validation = df.loc[(df.index >= train_date_threshold) & (df.index < validation_date_threshold)]
    test = df.loc[df.index >= validation_date_threshold]

    if describe:
        #Print time range
        print("start_dt:", df['time_utc'].min(), "\nend_dt:", df['time_utc'].max(), "\nnumber_days:", df['time_utc'].max() - df['time_utc'].min(), "\n")

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
#    * Multiple time series features --> [`reading_count`,`methane_mixing_ratio_bias_corrected_mean`,`dist_away_2_mean`,`qa_val_mean`] + `weather` + `oil wells`
#         * https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1 
#         * https://colab.research.google.com/gist/onesamblack/fae25efdcd82ba208c453bafeba86d3c/lstm_autoencoder.ipynb
#         * Window over last 7 days --> predict on next day. 
#         * Autoencoder **not** making as much sense, because we are recreating the input. We don't want to recreate all the other features....
#             * Input Shape (None, 7, 4)
#             * Output Shape (None, 7, 1)  
#             * As of now shape of 1 works. Because Y variable is dimension 1. but doesn't make sense...w/ Autoencoder
#             * I would have to make last dense layer back to 4, make our Y variable also of dimension 4, but that is recreating all the features... not exactly what we want either
#

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

# # Univariate Model

# ### Functions for Univariate Model

# +
####################################################################
# Function to standard scaler the data
####################################################################
def standardize_data(train, validation, test, feature_cols, describe=False):

    train_input = train[feature_cols]
    val_input = validation[feature_cols]
    test_input = test[feature_cols]

    mm_scaler = MinMaxScaler()
    mm_scaler = mm_scaler.fit(train_input)

    train_scaled = mm_scaler.transform(train_input)
    val_scaled = mm_scaler.transform(val_input)
    test_scaled = mm_scaler.transform(test_input)

    train_features = train_scaled
    val_features = val_scaled
    test_features = test_scaled

    if describe:
        print("train:", train_features.shape)
        print("val:", val_features.shape)
        print("test:", test_features.shape)

    return train_scaled, val_scaled, test_scaled, mm_scaler


####################################################################
# Function to run univariate neural network
####################################################################
def lstm_uni(trainX, trainY, valX, valY, plot=False):
    
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
    model.summary()

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


def calculate_loss(model, trainX, valX):
    
    #Predict model and calculate MSE
    X_train_pred = model.predict(trainX)
    X_val_pred = model.predict(valX)

    train_mse_loss = np.mean(np.square(X_train_pred- trainX), axis=1)
    val_mse_loss = np.mean(np.square(X_val_pred- valX), axis=1)    

    return train_mse_loss, val_mse_loss, X_train_pred, X_val_pred


####################################################################
# Functions to plot
####################################################################

def plotting_distplot(train_mse_loss, val_mse_loss):
    
    #MSE LOSS
#     print("Train MSE loss distribution")
    sns.distplot(train_mse_loss, bins=50, kde=True)
#     print("Validation MSE loss distribution")

    sns.distplot(val_mse_loss, bins=50, kde=True)
    plt.legend(labels=["TrainMSE","ValMSE"])


def model_analysis_plots(region, train_mse_loss, ANOMALY_THRESHOLD, val_score_df, validation, val_scaled, val_anomalies, mm_scaler):
    
    #combined plots
    fig, axs = plt.subplots(3, 1, figsize=(10,15))  #specify how many sub - plots in figure

    plt. subplots_adjust(hspace=0.5)   #spacing between each subplot
    titles_font_size = 12
    ticks_font_size = 8
    fig.suptitle("LSTM AE Univariate Model at Region {}".format(region))  #title for entire subplot

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
    plt.setp(axs[2].get_xticklabels(), fontsize=ticks_font_size, rotation=25)#, horizontalalignment="left")
    plt.setp(axs[2].get_yticklabels(), fontsize=ticks_font_size)#,  horizontalalignment="left")
    axs[2].set_title("Validation Methane", fontsize=titles_font_size)
    axs[2].plot(validation[window_length:].index,  mm_scaler.inverse_transform(val_scaled[window_length:]),  label = 'methane_mixing_ratio_bias_corrected_mean' )
    axs[2].scatter(
        val_anomalies.index,
        val_anomalies.methane_mixing_ratio_bias_corrected_mean,
        color = sns.color_palette()[3],
        s=10,
        label='anomaly')
    axs[2].legend(fontsize=titles_font_size)

    fig.savefig("./figures/univariate/lstmae_univariate_region{}".format(str(region)))

    
####################################################################
# Functions for Anomaly Detection
####################################################################

def anomaly(train_mse_loss, val_mse_loss, train, validation, test):

    upper,lower = np.percentile(train_mse_loss,[75,25])
    ANOMALY_THRESHOLD = 7*(upper-lower)
#     ANOMALY_THRESHOLD = 0.04

    val_score_df = pd.DataFrame(index=validation[window_length:].index)
    val_score_df['loss'] = val_mse_loss

    val_score_df['threshold'] = ANOMALY_THRESHOLD
    val_score_df['anomaly'] = val_score_df.loss > val_score_df.threshold
    val_score_df['methane_mixing_ratio_bias_corrected_mean'] = validation[window_length:].methane_mixing_ratio_bias_corrected_mean
    val_score_df['reading_count'] = validation[window_length:].reading_count
    val_score_df['qa_val_mean'] = validation[window_length:].qa_val_mean
    val_anomalies = val_score_df[val_score_df.anomaly]
    
    return val_score_df, val_anomalies, ANOMALY_THRESHOLD


# +
#load data
regions = [x for x in range(1,17)]
# regions = [5]

start=time.time()

for region in regions:

    df, train, validation, test = load_data(region)

    #parameters:
    feature_cols = ['methane_mixing_ratio_bias_corrected_mean']
    window_length = 7
    batch_size = 32
    num_features = len(feature_cols)
    epochs = 50

    #standardize data
    train_scaled, val_scaled, test_scaled, mm_scaler = standardize_data(train, validation, test, feature_cols)

    #generate trainX and trainY
    num_feats_train, trainX, trainY = generate_datasets(train_scaled, window_length)
    num_feats_val, valX, valY = generate_datasets(val_scaled, window_length)
    num_feats_test, testX, testY = generate_datasets(test_scaled, window_length)
    assert num_feats_train == num_feats_test == num_feats_val

    #Run LSTM univariate model and plot
    model, history = lstm_uni(trainX, trainY, valX, valY)
    train_mse_loss, val_mse_loss, X_train_pred, X_val_pred = calculate_loss(model, trainX, valX)

    #plot MSE for Train and Validation
    plotting_distplot(train_mse_loss, val_mse_loss)
    val_score_df, val_anomalies, ANOMALY_THRESHOLD = anomaly(train_mse_loss, val_mse_loss, train, validation, test)
    model_analysis_plots(region, train_mse_loss, ANOMALY_THRESHOLD, val_score_df, validation, val_scaled, val_anomalies, mm_scaler)

end = time.time()
print("TIME: {time:.2f} secs".format(time=(end-start)))


# -

# # Multivariate AutoEncoder - Manually Windowed

# ### Functions for Multivariate Model

# +
####################################################################
# Function to standard scaler the data
####################################################################
def standardize_data(train, validation, test, feature_cols, describe=False):

    train_input = train[feature_cols]
    val_input = validation[feature_cols]
    test_input = test[feature_cols]

    mm_scaler = MinMaxScaler()
    mm_scaler = mm_scaler.fit(train_input)

    train_scaled = mm_scaler.transform(train_input)
    val_scaled = mm_scaler.transform(val_input)
    test_scaled = mm_scaler.transform(test_input)

    train_features = train_scaled
    val_features = val_scaled
    test_features = test_scaled

    if describe:
        print("train:", train_features.shape)
        print("val:", val_features.shape)
        print("test:", test_features.shape)

    return train_scaled, val_scaled, test_scaled, mm_scaler


####################################################################
# Function to run multivariate neural network
####################################################################
def lstm_multi(trainX, trainY, valX, valY, plot=False):
    
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
    model.summary()

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


def calculate_loss(feature, model, trainX, valX):
    #Predict model and calculate MSE of the feature (0th feature = methane)
    feature_num = feature_number[feature]
    X_train_pred = model.predict(trainX)[:,:,feature_num]
    X_val_pred = model.predict(valX)[:,:,feature_num]
    
    trainX_methane = trainX[:,:,feature_num]
    valX_methane = valX[:,:,feature_num]    
    
    train_mse_loss = np.mean(np.square(X_train_pred -  trainX_methane), axis=1) 
    val_mse_loss = np.mean(np.square(X_val_pred -  valX_methane), axis=1)
    
    return train_mse_loss, val_mse_loss, X_train_pred, X_val_pred

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

def anomaly(train_mse_loss, val_mse_loss, train, validation, test):

    upper,lower = np.percentile(train_mse_loss,[75,25])
    ANOMALY_THRESHOLD = 5*(upper-lower)
#     ANOMALY_THRESHOLD = 0.04

    val_score_df = pd.DataFrame(index=validation[window_length:].index)
    val_score_df['loss'] = val_mse_loss
    val_score_df['threshold'] = ANOMALY_THRESHOLD
    val_score_df['anomaly'] = val_score_df.loss > val_score_df.threshold
    
    for feature in feature_cols:
        val_score_df[feature] = validation[window_length:][feature]
        
    val_anomalies = val_score_df[val_score_df.anomaly]
    
    return val_score_df, val_anomalies, ANOMALY_THRESHOLD


# +
#select region and features
regions = [x for x in range(1,17)]
# regions = [5]

#parameters:
feature_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count', 'qa_val_mean',
                'qa_val_mode',
               'air_pressure_at_mean_sea_level_mean',
               'air_temperature_at_2_metres_mean', 'air_temperature_at_2_metres_1hour_Maximum_mean', 'air_temperature_at_2_metres_1hour_Minimum_mean',
               'dew_point_temperature_at_2_metres_mean',
               'eastward_wind_at_100_metres_mean', 'eastward_wind_at_10_metres_mean',
               'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean',
               'lwe_thickness_of_surface_snow_amount_mean',
               'northward_wind_at_100_metres_mean', 'northward_wind_at_10_metres_mean',
               'precipitation_amount_1hour_Accumulation_mean', 'snow_density_mean',
               'surface_air_pressure_mean']

# feature_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count', 'qa_val_mean']

feature_number = dict()
number=0
for feature in feature_cols:
    feature_number[feature] = number
    number+=1

start=time.time()
    
for region in regions:
    
    df, train, validation, test = load_data(region)
    df=df.dropna()                    #NEED TO DROP ROWS WITH NA VALUES :(
    train=train.dropna()
    validation=validation.dropna()
    test=test.dropna()

    window_length = 7
    batch_size = 32
    num_features = len(feature_cols)
    epochs = 50

    #standardize data
    train_scaled, val_scaled, test_scaled, mm_scaler = standardize_data(train, validation, test, feature_cols)

    #generate trainX and trainY
    num_feats_train, trainX, trainY = generate_datasets(train_scaled, window_length)
    num_feats_val, valX, valY = generate_datasets(val_scaled, window_length)
    num_feats_test, testX, testY = generate_datasets(test_scaled, window_length)
    assert num_feats_train == num_feats_test == num_feats_val
    
    #Run LSTM univariate model and plot
    model, history = lstm_multi(trainX, trainY, valX, valY)

    
    for feature in feature_cols:
        #Predict MSE's:
        train_mse_loss, val_mse_loss, X_train_pred, X_val_pred = calculate_loss(feature, model, trainX, valX)

        #plot MSE for Train and Validation
#         plotting_distplot(train_mse_loss, val_mse_loss, feature)
        val_score_df, val_anomalies, ANOMALY_THRESHOLD = anomaly(train_mse_loss, val_mse_loss, train, validation, test)
        model_analysis_plots(region, train_mse_loss, ANOMALY_THRESHOLD, val_score_df, validation, val_scaled, val_anomalies, mm_scaler, feature, save=True)

end=time.time()
print("TIME: {time:.2f} secs".format(time=(end-start)))

# -


