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

# ### MATPLOTLIB - Plot Functions

# +

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
    feature_num = feature_number_map[feature]
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
        feature_num = feature_number_map[feature]
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

# ### Load Data

df, train, val, test = load_all_zone_data()



# ### Track Everything

# + jupyter={"outputs_hidden": true}
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

#Connect to S3 default profile
s3 = boto3.client('s3')
# -

# ### Write out to Pickle Files

## FIRST YOU NEED TO MAKE THIS FOLDER
today_dt = datetime.today().strftime('%Y%m%d')
f'zone_artifacts_{today_dt}'

# +
#### Dictionaries ####


# feature_loss_tracker
# df_tracker
# model_metrics_tracker

with open(f'zone_artifacts_{today_dt}/feature_loss_tracker.pickle', 'wb') as handle:
    pickle.dump(feature_loss_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'zone_artifacts_{today_dt}/df_tracker.pickle', 'wb') as handle:
    pickle.dump(df_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'zone_artifacts_{today_dt}/model_metrics_tracker.pickle', 'wb') as handle:
    pickle.dump(model_metrics_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)


##LOAD PICKLE FILE
# with open(f'zone_artifacts_{today_dt}/df_tracker.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# print a == b
# -

# ### Put on S3

# +
# import subprocess
# subprocess.check_output(['aws','s3','cp', '--recursive', 'zone_artifacts_20211101' , 's3://methane-capstone/models/autoencoder/zone_model_artifacts/'])
# -



# ### Analysis - MATPLOTLIB
#

# +
# # plot MSE for Train and Validation

# zone = 5
# feature = 'methane_mixing_ratio_bias_corrected_mean'
# train_mse_loss = feature_loss_tracker[zone]['train'][feature]['train_mse_loss']
# val_mse_loss = feature_loss_tracker[zone]['val'][feature]['val_mse_loss']
# train_zone = df_tracker[zone]['train_zone']
# val_zone = df_tracker[zone]['val_zone']
# val_scaled = df_tracker[zone]['val_scaled']
# scaler = df_tracker[zone]['scaler']


# plotting_distplot(train_mse_loss,
#                   val_mse_loss,
#                   feature
#                  )
# -

# ### Look at Anomalies

# +

# val_score_df, val_anomalies, ANOMALY_THRESHOLD = anomaly(train_mse_loss, val_mse_loss, train_zone, val_zone)

# ### Plot Analysis Plots
# model_analysis_plots(zone, 
#                      train_mse_loss,
#                      ANOMALY_THRESHOLD,
#                      val_score_df, 
#                      val_zone,
#                      val_scaled, 
#                      val_anomalies,
#                      scaler,
#                      feature,
#                      save=False)
# -





# ### Analysis - ALTAIR

# +
def get_anomaly_threshold(mse_loss):
    upper,lower = np.percentile(mse_loss,[75,25])
    ANOMALY_THRESHOLD = 5*(upper-lower)
    return ANOMALY_THRESHOLD


def apply_threshol(anom_thresh, data_frame, mse_loss):

    scored_df = pd.DataFrame(index=data_frame[window_length:].index)
    scored_df['loss'] = mse_loss
    scored_df['threshold'] = anom_thresh
    scored_df['anomaly'] = scored_df.loss > scored_df.threshold
    
    for feature in feature_cols:
        scored_df[feature] = data_frame[window_length:][feature]
        
    return scored_df



# -

zones

feature_cols

# ### Build Final Data Frames For Visuals

# +
# plot MSE for Train and Validation
final_dataframes = {key: {'train':None, 'val': None, 'test': None} for key in zones}

for zone in zones:
    for split in ['train', 'val', 'test']:
            
        cur_zone_df = df_tracker[zone][f'{split}_zone']
        scored_df = pd.DataFrame(index=cur_zone_df[window_length:].index)
    
        for feature in feature_cols:
            
            ### WE MIGHT HAVE TO FIGURE OUT THE COLOR BUCKETS HERE ###
            
            train_mse_loss = feature_loss_tracker[zone]['train'][feature][f'train_mse_loss']
            mse_loss = feature_loss_tracker[zone][split][feature][f'{split}_mse_loss']
            
            anom_thresh = get_anomaly_threshold(train_mse_loss)

            
            scored_df[feature] = cur_zone_df[window_length:][feature]
            scored_df[f'{feature}_loss'] = mse_loss
            scored_df[f'{feature}_threshold'] = anom_thresh
            scored_df[f'{feature}_anomaly'] = scored_df[f'{feature}_loss'] > scored_df[f'{feature}_threshold']

        final_dataframes[zone][split] = scored_df
        

# -

# ### Sample check the data

# +
print(final_dataframes[1]['train'].shape)
print(final_dataframes[1]['val'].shape)
print(final_dataframes[1]['test'].shape)
print()

print(final_dataframes[2]['train'].shape)
print(final_dataframes[2]['val'].shape)
print(final_dataframes[2]['test'].shape)
print()
# -



# ## Our Colors

# +
#Methane
ven_red = '#C91414'
cad_ed ='#E3071D'
amber ='#FF7E01'
flu_orange ='#FFBE00'
bud_green ='#75AD6F'
dark_green ='#1D7044'


#Weather + etc.
carrot_orange ='#DD8420'
gold_yellow ='#DFA829'
bone = '#DAD5C7'
wel_blue= '#769DB2'
steel_blue ='#497FA8'
dazzle_blue = '#2E5791'


# -

# ## Loss Over Time

def plot_multi_loss(zone, split, feat_sub_cols):

    '''
    zone = 1-16
    split = ['train', 'val', 'test']
    feat_sub_cols = [<LIST OF FEATURES>]
    '''
    
    #Data
    df_viz = final_dataframes[zone][split]
    
    #Get all columns requested for analysis
    sub_cols = []
    for feat_col in feat_sub_cols:
        for col in df_viz.columns:
            if feat_col in col:
                sub_cols.append(col)    
    df_viz = df_viz[sub_cols].reset_index()
    
    
    #Create Concatenated Charts
    chart = alt.vconcat(data=df_viz)
    row = alt.hconcat()
    split_row = 1
    
    for i in range(1, len(feat_sub_cols)+1):
        
        #Flag to include timestamp ticks
        show_ts = False #i == len(feat_sub_cols)
        
        #Features
        feature = feat_sub_cols[i-1]
        feat_loss_col = feature+'_loss'
        feat_anom_col = feature+'_anomaly'
        feat_thresh_col = feature+'_threshold'
    
        #Loss Points
        points = alt.Chart(df_viz).mark_circle(size=50, tooltip=True).encode(
            x = alt.X('time_utc:O', axis=alt.Axis(labels=show_ts)),
            y= alt.Y(f'{feat_loss_col}:Q'),    
            color=alt.Color(f'{feat_anom_col}:N', title='Anomaly',
                               scale=alt.Scale(domain=[False, True],
                                range=[dark_green, ven_red]))
        )

        #Loss Line
        line = alt.Chart(df_viz).mark_line(color='lightgrey').encode(
            x = alt.X('time_utc:O', axis=alt.Axis(labels=show_ts)),
            y = alt.Y(f'{feat_loss_col}:Q', title=feat_loss_col)
        )

        #Anom Thresh
        rule = alt.Chart(df_viz).mark_rule(color=flu_orange, strokeDash=[10], tooltip =True).encode(
                y=alt.Y(f'{feat_thresh_col}:Q'),
                size=alt.value(2)
        )

        row |=  (points + line + rule).properties(width = 1100, height=200)

        if i % split_row == 0:
            chart &= row
            row = alt.hconcat()
        
    return chart.properties(title=f"Zone #{zone} Loss").configure_title(
                                                            fontSize=20,
                                                            font='Courier',
                                                            anchor='middle',
                                                            color='gray',
                                                        )


feature_cols

# ### Plot Loss by Zone

cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count','qa_val_mean']
plot_multi_loss(1, 'val', cols)



# ### Plot Features by Zone

def plot_multi_time_series(zone, split, feat_sub_cols, time_agg='', agg=''):

    #Either both `aggs` or no `agg`
    if (time_agg and not agg) or (not time_agg and agg):
        raise Exception("Must be all or none")

    #Data
    df_viz = final_dataframes[zone][split]
    
    #Get all columns requested for analysis
    sub_cols = []
    for feat_col in feat_sub_cols:
        for col in df_viz.columns:
            if feat_col in col and 'thresh' not in col:
                sub_cols.append(col)
    df_viz = df_viz[sub_cols].reset_index()

    #Time Selection
    brush = alt.selection(type='interval', encodings=['x'])
    
    x_encoding = f'{time_agg}(time_utc):O' if time_agg else f'time_utc:O'
    
    #Pick Split
    if agg:
        split_row = 2
    else:
        split_row = 1

    #Create Concatenated Charts
    chart = alt.vconcat(data=df_viz)
    row = alt.hconcat()
    for i in range(1, len(feat_sub_cols)+1):
        y_encoding = feat_sub_cols[i-1]
        
        if agg:
            #Loss Line
            row |= alt.Chart(df_viz).mark_line(point={
                                                  "filled": False,
                                                  "fill": "white"
                                                }, tooltip=True).encode(
                                                    x = alt.X(x_encoding),
                                                    y = alt.Y(f'{agg}({y_encoding}):Q', 
                                                             scale=alt.Scale(zero=False)),
                                                   color=alt.condition(brush, alt.value(bud_green), alt.value('lightgray'))
                                                    ).add_selection(brush)
        
        else:
            
            feat_anom_col = y_encoding+'_anomaly'
            
            #Loss Points
            points = alt.Chart(df_viz).mark_circle(size=50, tooltip=True).encode(
                x = alt.X(x_encoding, axis=alt.Axis(labels=False)),
                y = alt.Y(f'{y_encoding}:Q', 
                         scale=alt.Scale(zero=False)),
                color=alt.condition(brush, 
                                    alt.Color(f'{feat_anom_col}:N',
                                              title='Anomaly',
                                              scale=alt.Scale(
                                                  domain=[False, True],
                                                  range=[dark_green, ven_red])),
                                    alt.value('lightgray'))
            ).add_selection(brush)

            #Loss Line
            line = alt.Chart(df_viz).mark_line(color='lightgrey').encode(
                x = alt.X(x_encoding, axis=alt.Axis(labels=False)),
                y = alt.Y(f'{y_encoding}:Q', 
                         scale=alt.Scale(zero=False)),
            )

            row |= (points + line).properties(width = 1100, height=200)
            
        if i % split_row == 0:
            chart &= row
            row = alt.hconcat()

    if agg:
        return chart
    else:
        return chart



# +
feat_sub_cols = ['methane_mixing_ratio_bias_corrected_mean','reading_count',
               'air_temperature_at_2_metres_mean', 'eastward_wind_at_10_metres_mean'
              ]


plot_multi_time_series(3, 'train', feat_sub_cols, time_agg='yearmonth', agg='mean')
# -


