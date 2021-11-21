# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (Data Science)
#     language: python
#     name: python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/datascience-1.0
# ---

# # LSTM Autoencoder Model Analysis - New Mexico  
#
# This notebook uses California Climate Zone pretrained models to predict anomalies on a specific region of Carlsbad, New Mexico data.  The leak occured in 2019/10/23 per this paper: http://www.sci.utah.edu/publications/Foo2021a/1-s2.0-S0034425721002947-main.pdf  
#
# This notebook generates plots of the anomaly prediction and saves the plots locally on sagemaker.  

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



# +
#####################################
# Function to load data, given region
#####################################
def load_all_zone_data(describe=True):

    #Read in Data
    s3_file_path = 's3://methane-capstone/data/testsite3/combined-raw-data/combined-raw-bcj.parquet'
    df = pd.read_parquet(s3_file_path)
    df['time_utc'] = pd.to_datetime(df['time_utc']).dt.date.astype('datetime64[ns]')
    df_reduced = df.groupby('time_utc').agg({'methane_mixing_ratio_bias_corrected': "mean",
                                             'methane_mixing_ratio': ["count"],
                                             'air_pressure_at_mean_sea_level': ["mean"],
                                             'eastward_wind_at_100_metres': ["mean"],
                                             'northward_wind_at_100_metres': ["mean"],                   
                                             'air_temperature_at_2_metres': ["mean"],
                                             'surface_air_pressure': ["mean"],
                                             'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation' : ["mean"],
                                             'precipitation_amount_1hour_Accumulation': ["mean"],
                                             'dew_point_temperature_at_2_metres': ["mean"],
                                              })

    #Flatten MultiIndex
    df_reduced.columns = ['_'.join(col) for col in df_reduced.columns.values]
    df_reduced = df_reduced.reset_index()
    df_reduced = df_reduced.rename(columns={"methane_mixing_ratio_count": "reading_count"})
    df = df_reduced
    df['BZone'] = 1
    df.set_index(['time_utc'], inplace=True)

    train_date_threshold = '2019-10-30'
#     validation_date_threshold = '2021-06-01'   #Original threshold but our leak is on 6/4/2021
    validation_date_threshold = '2019-10-30'

    train = df.loc[df.index > train_date_threshold]
    validation = df.loc[(df.index <= train_date_threshold)]
    test =df.loc[(df.index <= train_date_threshold)]
    
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

    import pickle
    standard_scaler_name = 'ScalerModel_20211120'                              
    scaler = pickle.load(open(f'/root/methane/models/autoencoder/{standard_scaler_name}.pkl','rb'))

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
                         validation, val_scaled, val_anomalies, mm_scaler, feature, pretrained_model='', pretrained=False , save=False):
    
    #combined plots
    fig, axs = plt.subplots(3, 1, figsize=(10,15))  #specify how many sub - plots in figure

    plt. subplots_adjust(hspace=0.5)   #spacing between each subplot
    titles_font_size = 12
    ticks_font_size = 8
    fig.suptitle("LSTM AE Trained Model, Feature {}".format(region, feature))  #title for entire subplot
    if pretrained:
        fig.suptitle("LSTM AE Pretrained Model ({}), Feature {}".format(pretrained_model,region, feature))  #title for entire subplot
    
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
        fig.savefig("./figures/newmexico/lstmae_pretrained_on_{}".format(pretrained_model))
    
#     fig.clear(True)
    
####################################################################
# Functions for Anomaly Detection
####################################################################

def anomaly(train_mse_loss, other_mse_loss, train, other_data):

    upper,lower = np.percentile(train_mse_loss,[75,25])
    ANOMALY_THRESHOLD = 3*(upper-lower)

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
# Load pretrained LSTMAE model and Standard Scaler to local sage maker
import subprocess
subprocess.check_output(['aws','s3','cp', '--recursive', 's3://methane-capstone/models/autoencoder/pretrained/' ,'/root/methane/models/autoencoder/' ])

# +
#select region and features
zones = [1]  #all of this dataset in New Mexico is zone 1 for simplicity

# Track predictions and losses for analysis across different features
feature_loss_tracker = {key: {'train':{}, 'val':{}, 'test':{}} for key in zones}

#Track all the data frames, raw and scaled
df_tracker = {}

#Track all the metrics from each model training cycle
model_metrics_tracker = {}

drop = False
pretrained=True
model_names = ['LSTMAE_Zone1_20211120',
                 'LSTMAE_Zone2_20211120',
                 'LSTMAE_Zone3_20211120',
                 'LSTMAE_Zone4_20211120',
                 'LSTMAE_Zone5_20211120',
                 'LSTMAE_Zone6_20211120',
                 'LSTMAE_Zone7_20211120',
                 'LSTMAE_Zone8_20211120',
                 'LSTMAE_Zone9_20211120',
                 'LSTMAE_Zone10_20211120',
                 'LSTMAE_Zone11_20211120',
                 'LSTMAE_Zone12_20211120',
                 'LSTMAE_Zone13_20211120',
                 'LSTMAE_Zone14_20211120',
                 'LSTMAE_Zone15_20211120',
                 'LSTMAE_Zone16_20211120']

#make local directory to store results
local_path = '/root/methane/models/autoencoder/john/figures/newmexico/'
try:
    os.makedirs(local_path)
except:
    print("")
    
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

for model_name in model_names:
    
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

        if pretrained == False: 
            print("training model")
            #Run LSTM Multivariate model and plot
            model, history =  lstm_multi(trainX, trainY, valX, valY, window_length, num_features, batch_size, epochs, plot=False)
            model_metrics_tracker[zone] = history.history

        if pretrained == True:
            print("loading modeling")
            #Load model
            model = tf.keras.models.load_model(f'/root/methane/models/autoencoder/{model_name}.h5', compile=False)


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



    # LOOK AT ANOMALIES
    feature = 'methane_mixing_ratio_bias_corrected_mean'
    val_score_df, val_anomalies, ANOMALY_THRESHOLD = anomaly(train_mse_loss, val_mse_loss, train_zone, val_zone)

    ### Plot Analysis Plots
    model_analysis_plots(zone, 
                         train_mse_loss,
                         ANOMALY_THRESHOLD,
                         val_score_df, 
                         val_zone,
                         val_scaled, 
                         val_anomalies,
                         scaler,
                         feature, 
                         pretrained_model=model_name, 
                         pretrained=True,
                         save=True)
# -
# # DID NOT USE ANY CELLS BELOW TO CREATE PRETRAINED MODEL - JOHN

# + [markdown] jupyter={"source_hidden": true}
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

# + [markdown] jupyter={"source_hidden": true}
# ### Write out to Pickle Files

# + jupyter={"source_hidden": true}
## FIRST YOU NEED TO MAKE THIS FOLDER
today_dt = datetime.today().strftime('%Y%m%d')
f'zone_artifacts_{today_dt}'

# + jupyter={"source_hidden": true}
#### Dictionaries ####

import boto3
import pickle

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

# + [markdown] jupyter={"source_hidden": true}
# ### Put on S3

# + jupyter={"source_hidden": true}
# import subprocess
# subprocess.check_output(['aws','s3','cp', '--recursive', 'zone_artifacts_20211101' , 's3://methane-capstone/models/autoencoder/zone_model_artifacts/'])
# + jupyter={"source_hidden": true}

# -


# ### Import PKL From S3

# + jupyter={"source_hidden": true}
import boto3
import pickle

bucket = 'methane-capstone'
flt_path = 'models/autoencoder/zone_model_artifacts/testsites/newmexico_feature_loss_tracker.pickle'
dft_path = 'models/autoencoder/zone_model_artifacts/testsites/newmexico_df_tracker.pickle'
mmt_path = 'models/autoencoder/zone_model_artifacts/testsites/newmexico_model_metrics_tracker.pickle'


#Connect to S3 default profile
s3client = boto3.client('s3')


feature_loss_tracker = pickle.loads(s3client.get_object(Bucket=bucket, Key=flt_path)['Body'].read())
df_tracker = pickle.loads(s3client.get_object(Bucket=bucket, Key=dft_path)['Body'].read())
model_metrics_tracker = pickle.loads(s3client.get_object(Bucket=bucket, Key=mmt_path)['Body'].read())

feature_cols = list(feature_loss_tracker[1]['train'].keys())
feature_cols
# -

# ### Analysis - MATPLOTLIB
#

# + jupyter={"source_hidden": true}
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

# ### Analysis - ALTAIR

# + jupyter={"source_hidden": true}
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




# + jupyter={"source_hidden": true}
zones

# + jupyter={"source_hidden": true}
feature_cols
# -

# ### Build Final Data Frames For Visuals

# + jupyter={"source_hidden": true}
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

# + jupyter={"source_hidden": true}
print(final_dataframes[1]['train'].shape)
print(final_dataframes[1]['val'].shape)
print(final_dataframes[1]['test'].shape)
print()

print(final_dataframes[2]['train'].shape)
print(final_dataframes[2]['val'].shape)
print(final_dataframes[2]['test'].shape)
print()
# + jupyter={"source_hidden": true}

# -


# ## Our Colors

# + jupyter={"source_hidden": true}
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

# + jupyter={"source_hidden": true}
set(final_dataframes[1]['train'].columns) - set(final_dataframes[1]['val'].columns)

# + jupyter={"source_hidden": true}
feature_cols

# + jupyter={"source_hidden": true}


# + jupyter={"source_hidden": true}


# + jupyter={"source_hidden": true}
def plot_multi_loss(zone, split, feat_sub_cols):

    '''
    zone = 1-16
    split = ['train', 'val', 'test']
    feat_sub_cols = [<LIST OF FEATURES>]
    '''
    
    #Data
    df_viz = None
    if split:
        df_viz = final_dataframes[zone][split]

    else:
        df_viz = pd.concat([final_dataframes[zone]['train'],final_dataframes[zone]['val'],final_dataframes[zone]['test']])
    
    #Get all columns requested for analysis
    sub_cols = []
    for feat_col in feat_sub_cols:
        for col in df_viz.columns:
            if feat_col in col:
                sub_cols.append(col)    
    df_viz = df_viz[sub_cols].reset_index()
    
    #Time Selection
    brush = alt.selection(type='interval', encodings=['x'])
    
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
            color=alt.condition(brush, 
                                    alt.Color(f'{feat_anom_col}:N',
                                              title='Anomaly',
                                              scale=alt.Scale(
                                                  domain=[False, True],
                                                  range=[dark_green, ven_red])),
                                    alt.value('lightgray'))
            ).add_selection(brush)
        
        
# alt.Color(f'{feat_anom_col}:N', title='Anomaly',
#                                scale=alt.Scale(domain=[False, True],
#                                 range=[dark_green, ven_red]))
#         ).add_selection(brush)

        
        
        
    
        
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
# -

# ### FINAL FEATURE LIST 
#
# * 'methane_mixing_ratio_bias_corrected_mean'
# * 'reading_count'
# * 'air_pressure_at_mean_sea_level_mean'
# * 'eastward_wind_at_100_metres_mean'
# * 'northward_wind_at_100_metres_mean'
# * 'air_temperature_at_2_metres_mean'
# * 'surface_air_pressure_mean'
# * 'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation_mean'
# * 'precipitation_amount_1hour_Accumulation_mean' 
# * 'dew_point_temperature_at_2_metres_mean'

# ### Plot Loss by Zone

# + jupyter={"source_hidden": true}
plot_multi_loss(1, '', feature_cols[:3])

# + jupyter={"source_hidden": true}


# + jupyter={"source_hidden": true}
boom = final_dataframes[1]['train'][['methane_mixing_ratio_bias_corrected_mean',  'eastward_wind_at_100_metres_mean']]

# + jupyter={"source_hidden": true}
plt.scatter(x=boom['methane_mixing_ratio_bias_corrected_mean'] , y =boom[ 'eastward_wind_at_100_metres_mean'])

# + jupyter={"source_hidden": true}
plot_multi_time_series(14, 'val', feature_cols)# time_agg='yearmonth', agg='mean')

# + jupyter={"source_hidden": true}
examp_chart = plot_multi_loss(1, 'val', feature_cols)

# + jupyter={"source_hidden": true}
examp_chart.save('loss_compare.json')

# + jupyter={"source_hidden": true}
examp_chart


# -

# ### Plot Features by Zone

# + jupyter={"source_hidden": true}
def plot_multi_time_series(zone, split, feat_sub_cols, time_agg='', agg=''):

    #Either both `aggs` or no `agg`
    if (time_agg and not agg) or (not time_agg and agg):
        raise Exception("Must be all or none")

    #Data
    df_viz = None
    if split:
        df_viz = final_dataframes[zone][split]
    else:
        df_viz = pd.concat([final_dataframes[zone]['train'],final_dataframes[zone]['val'],final_dataframes[zone]['test']])
    
    
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
            #Agg Value Line
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
            
            #Value Points
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

            #Value Line
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


# + jupyter={"source_hidden": true}
len(feature_cols)

# + jupyter={"source_hidden": true}
plot_multi_time_series(1, '', feature_cols, time_agg='yearmonth', agg='mean')
