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

# +
# # !conda install -c plotly plotly==3.10.0 --yes
# # !conda install -c conda-forge fbprophet --yes
# -

# # Modeling Sentinel 5p Data with FB Prophet
#
# Each model is x1 California Climate Zone
#
#

# # Import Packages and Data

# +
#Packages 
from fbprophet import Prophet
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import plotly.express as px

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid'] = False

# +
# Read in Data
bucket = 'methane-capstone'
subfolder = 'data/combined-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='data-zone-combined.parquet.gzip'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)

#Read
df = pd.read_parquet(data_location)

#Convert date into yyyy-mm-dd format
# df['date_formatted'] = pd.to_datetime(df['time_utc'], format='%y-%m-%d h%:m%:s%').dt.strftime('%Y-%m-%d')
#Convert to correct Type
df['date_formatted'] = pd.to_datetime(df['time_utc'])

#Simple EDA
print("SHape: ", df.shape)
print("Types: ", df.dtypes)
df.head()

# +
# Read in High Quality Data
bucket = 'methane-capstone'
subfolder = 'data/combined-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='data-zone-combined-highqual.parquet.gzip'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)

#Read
df_highqual = pd.read_parquet(data_location)

#Convert date into yyyy-mm-dd format
# df['date_formatted'] = pd.to_datetime(df['time_utc'], format='%y-%m-%d h%:m%:s%').dt.strftime('%Y-%m-%d')
#Convert to correct Type
df_highqual['date_formatted'] = pd.to_datetime(df_highqual['time_utc'])

#Simple EDA
print("SHape: ", df_highqual.shape)
print("Types: ", df_highqual.dtypes)
df_highqual.head()


# -

# # Create FB Prophet Modeling and Graph Functions

def FBpropet(full_df, zone, changepoint_range=0.95, future_periods=273, anomaly_factor=1, standardize= True):

    """
    Inputs:  
        "full_df" formatted with two columns, 'ds' = datetime, 'y' = value
        
        "changepoint_range" for % of data, 0.95 will place potential chagnepoints 
        in the first 95% of the time series. higher changepoint_range = can overfit to train data
        
        "future_periods" for number of days to forecast future data (size of test data)
        higher future_periods = more forecasting
        
        "anomaly_factor" for labeling points outside of factor times uncertainty as anomalies, 
        higher anomaly_factor = less anomalies    
    
    Outputs:
        model = the model that was trained with the input dataset
        forecast = default forecasting dataframe from fb-prophet package, lots of helpful information
        results = simplied version of the forecast dataframe but includes column for anomaly or not
        anomaly_df = results dataframe but only rows that are anomalies    
    """
    
   
    #Split dataframe into train and test
    train = full_df[(full_df['ds']<'2021-01-01')]
    test = full_df[(full_df['ds']>'2020-12-31') & (full_df['ds']<'2021-06-02')]
    predict_df = full_df[(full_df['ds']<'2021-06-02')]

    
    if standardize:
        #Standardize Dataset
        scaler = StandardScaler()
        scaler = scaler.fit(train['y'].values.reshape(-1, 1))               #Fit to train data

        train_scaled = scaler.transform(train['y'].values.reshape(-1, 1))   #Transform train data to Train Data scale
        test_scaled = scaler.transform(test['y'].values.reshape(-1, 1))     #Transform test data to Train Data scale
        predict_df_scaled = scaler.transform(predict_df['y'].values.reshape(-1, 1))     #Transform test data to Train Data scale
      
        train_rows = train.shape[0]
        test_rows = test.shape[0]
        predict_df_rows = predict_df.shape[0]
        
        train['y'] = train_scaled.reshape(1,train_rows)[0]                       #Replace not standardized column
        test['y'] = test_scaled.reshape(1,test_rows)[0]                           #Replace not standardized column
        predict_df['y'] = predict_df_scaled.reshape(1,predict_df_rows)[0]         #Replace not standardized column
   
    # dictionary of parameters (values) for each zone (key)
    # parameters = [changepoint_prior_scale,	seasonality_prior_scale,	seasonality_mode]

    mp = {
        1:[0.5,    0.1,  'additive'],
        2:[0.5,    0.01, 'multiplicative'],
        3:[0.001,  0.1,  'additive'],
        4:[0.05,   0.1,  'additive'],
        5:[0.1,    0.1,  'additive'],
        6:[0.5,    1,    'additive'],
        7:[0.5,    0.1,  'additive'],    
        8:[0.1,    0.1,  'additive'],
        9:[0.5,    0.1,  'additive'],
        10:[0.5,   0.1,  'additive'],
        11:[0.5,   0.1,  'multiplicative'],
        12:[0.001, 0.1,  'additive'],
        13:[0.05,  0.1,  'additive'],
        14:[0.05,  0.1,  'additive'],
        15:[0.5,   10,   'additive'],
        16:[0.05,  0.01, 'additive'],
    }

    model = Prophet(changepoint_prior_scale=mp[zone][0],
                    seasonality_prior_scale=mp[zone][1],
                    seasonality_mode=mp[zone][2])          #Create Prophet Model. Default is 80% changepoint
    model.fit(train)                          #Fit on Training     

    #Forecasting
    if future_periods == 273:
        future_periods = test.shape[0]
    
    future = model.make_future_dataframe(periods=future_periods)  #Create Future dataframe for predicting historical and future dates
    forecast = model.predict(future)          #Forecast dataset

    #Create a new dataframe that has the forecasts plus the actual true value for that day
    results = pd.concat([predict_df.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower', 'yhat_upper']]],axis=1) 

    #OUTLIER DETECTION
    results['error'] = results['y'] - results['yhat']                      #error = true minus predict
    results['error2'] = results['error'] * results['error']                 #error2 = error*error
    results['uncertainty'] = results['yhat_upper'] - results['yhat_lower'] #unvertainty = range of prediction interval
    rmse = (sum(results['error2'].dropna()) / len(results['error2'].dropna())) ** 0.5
    
    #Look for outliers
    #Create anomaly column that will flag the row as an anomaly
    #If greater than ANOMALY_FACTOR times uncertainty value then it's an outlier
    results['anomaly'] = results.apply(lambda x: 'Yes' if(np.abs(x['error'])>anomaly_factor*x['uncertainty']) else 'No',axis=1)
    results = results.reset_index()
    anomaly_df = results[results.anomaly == 'Yes']
    
    return model, forecast, results, anomaly_df, train, test, rmse, scaler


def FBpropet_plot(scaler, model, forecast, results, anomaly_df, test, plot_title):
    
    """
    Inputs:
        model = the model that was trained with the input dataset
        forecast = default forecasting dataframe from fb-prophet package, lots of helpful information
        results = simplied version of the forecast dataframe but includes column for anomaly or not
        anomaly_df = results dataframe but only rows that are anomalies
        test = test data already formatted to two columns of "ds" and "y"
        plot_title = string to appear on top of plot
    
    Output:
        One plot
    """
# mm_scaler.inverse_transform(val_scaled[window_length:])[:,feature_num]

    forecast_inverse = forecast[['ds','trend', 'yhat_lower', 'yhat_upper', 'yhat']]
    forecast_inverse['trend'] = scaler.inverse_transform(forecast_inverse['trend'])
    forecast_inverse['yhat_lower'] = scaler.inverse_transform(forecast_inverse['yhat_lower'])
    forecast_inverse['yhat_upper'] = scaler.inverse_transform(forecast_inverse['yhat_upper'])
    forecast_inverse['yhat'] = scaler.inverse_transform(forecast_inverse['yhat'])
#     forecast_inverse
#     model.plot(forecast_inverse)

    #Plotting
    fig1 = model.plot(forecast)
    fig1 = plt.scatter(x=test.ds, y=test.y, c='b', marker = ".")
    fig1 = plt.legend(['Train Actuals', 'Forecast', '95% CI', 'Test Actuals'], fontsize=12, loc='upper left')
    fig1 = plt.xlabel("Date", size=18)
    fig1 = plt.ylabel("Methane Concentration (scaled)", size=18)
    fig1 = plt.xticks(fontsize=12)
    fig1 = plt.yticks(fontsize=12)
    fig1 = plt.title(plot_title, size=25)
    fig1 = plt.scatter(x=anomaly_df.ds, y=anomaly_df.y, c='r', s=50)
    fig1 = plt.axvline(datetime(2019, 1, 1))
    fig1 = plt.axvline(datetime(2020, 1, 1))
    fig1 = plt.axvline(datetime(2021, 1, 1))
    plt.savefig('images/{}.png'.format(plot_title))

    #Dark blue are yhat
    #Light blue is yhat confidence interval

    #plot the components
    #Top is the trend
    #Middle is weekly trend
    #Bottom is trend throughout the year
#     fig2 = model.plot_components(forecast)


# # Model_1: All Data per Zone

# +
zones = [x for x in range(1,17)]
rmse_list = []
all_data_anomalies = {}
for zone in zones:
    df_qual = df[df['BZone'] == zone ]
    # df_qual = df_qual[df_qual['qa_val_mean']>.4]


    #Average all methane readings for one day
    averaged_series = df_qual.groupby('date_formatted')['methane_mixing_ratio_mean'].mean()
    averaged_df = averaged_series.to_frame().reset_index()

    #Create new dataframe with the columns named as ds and y, as that is what Prophet requires
    methane_df = averaged_df.reset_index()[['date_formatted', 'methane_mixing_ratio_mean']].rename({'date_formatted':'ds', 
                                                               'methane_mixing_ratio_mean':'y'}, 
                                                              axis='columns')
    #Create Model and Forecast
    model, forecast, results, anomaly_df, train, test, rmse, scaler = FBpropet(methane_df, zone=1, changepoint_range=0.95, future_periods=273, anomaly_factor=1)
#     print("Zone_{} RMSE: {}".format(zone, rmse))
    rmse_list.append(rmse)
    
    #Plot
    plot_title = 'Zone_{}_Methane'.format(zone)
    FBpropet_plot(scaler, model, forecast, results, anomaly_df, test, plot_title)
    
    all_data_anomalies[zone] = anomaly_df
    
for i in zones:
    print("Zone_{} RMSE:  {}".format(i, rmse_list[i-1]))

# -

# # Model_2: High Quality Data Per Zone (High Quality Data Filtered After Pre Process)

df[['time_utc','qa_val_mean', 'qa_val_mode']].head()  #first row has 0.4 qa_val for both mode and mean
modequal_ZeroPtFour = df['qa_val_mode'].iloc[0]
meanqual_ZeroPtFour = df['qa_val_mean'].iloc[0]

print("mode: rows less than 0.4:", df[['time_utc', 'qa_val_mode']][df['qa_val_mode'] < modequal_ZeroPtFour].shape)
print("mode: rows equal to 0.4:", df[['time_utc', 'qa_val_mode']][df['qa_val_mode'] == modequal_ZeroPtFour].shape)
print("mode: rows greater than 0.4:", df[['time_utc', 'qa_val_mode']][df['qa_val_mode'] > modequal_ZeroPtFour].shape)
print("mean: rows less than 0.4:", df[['time_utc', 'qa_val_mean']][df['qa_val_mean'] < meanqual_ZeroPtFour].shape)
print("mean: rows equal to 0.4:", df[['time_utc', 'qa_val_mean']][df['qa_val_mean'] == meanqual_ZeroPtFour].shape)
print("mean: rows greater than 0.4:", df[['time_utc', 'qa_val_mean']][df['qa_val_mean'] > meanqual_ZeroPtFour].shape)
#Due to these distributions of quality numbers, qa_val_mode is ignored because most data is of quality 0.4
#qa_val_mean however has half the data greater than qa_val >  0.4, therefore use qa_val_mean > 0.4 as higher quality data

plt.hist(df['qa_val_mean'], bins=50)

# +
zones = [x for x in range(1,17)]
rmse_list = []
highqual_data_anomalies = {}


for zone in zones:
    df_qual = df[df['BZone'] == zone ]
    df_qual = df_qual[df_qual['qa_val_mean']>.4]


    #Average all methane readings for one day
    averaged_series = df_qual.groupby('date_formatted')['methane_mixing_ratio_mean'].mean()
    averaged_df = averaged_series.to_frame().reset_index()

    #Create new dataframe with the columns named as ds and y, as that is what Prophet requires
    methane_df = averaged_df.reset_index()[['date_formatted', 'methane_mixing_ratio_mean']].rename({'date_formatted':'ds', 
                                                               'methane_mixing_ratio_mean':'y'}, 
                                                              axis='columns')
    #Create Model and Forecast
    model, forecast, results, anomaly_df, train, test, rmse, scaler = FBpropet(methane_df, zone=1, changepoint_range=0.95, future_periods=273, anomaly_factor=1)
#     print("Zone_{} RMSE: {}".format(zone, rmse))
    rmse_list.append(rmse)

    #Plot
    plot_title = 'Zone_{}_Methane_HighQual'.format(zone)
    FBpropet_plot(scaler, model, forecast, results, anomaly_df, test, plot_title)
    
    highqual_data_anomalies[zone] = anomaly_df

for i in zones:
    print("Zone_{} RMSE:  {}".format(i, rmse_list[i-1]))

# -

## FIRST YOU NEED TO MAKE THIS FOLDER
today_dt = datetime.today().strftime('%Y%m%d')
f'zone_artifacts_{today_dt}'

# +
import boto3
import pickle

with open(f'zone_artifacts_{today_dt}/fbprophet_anomaly_highqualdata.pickle', 'wb') as handle:
    pickle.dump(highqual_data_anomalies, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'zone_artifacts_{today_dt}/fbprophet_anomaly_alldata.pickle', 'wb') as handle:
    pickle.dump(all_data_anomalies, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -

import subprocess
subprocess.check_output(['aws','s3','cp', '--recursive', 'zone_artifacts_20211118' , 's3://methane-capstone/models/fb-prophet/zone_model_artifacts/'])

# +
bucket = 'methane-capstone'
highqualdata = 'models/fb-prophet/zone_model_artifacts/fbprophet_anomaly_highqualdata.pickle'
alldata = 'models/fb-prophet/zone_model_artifacts/fbprophet_anomaly_alldata.pickle'


#Connect to S3 default profile
s3client = boto3.client('s3')

df_highqualdata = pickle.loads(s3client.get_object(Bucket=bucket, Key=highqualdata)['Body'].read())
df_alldata = pickle.loads(s3client.get_object(Bucket=bucket, Key=alldata)['Body'].read())

# -

df_alldata[1]

df_highqualdata[2]

# # Model_3: High Quality Data Per Zone (High Quality Data Filtered Before Pre Process)

# +
# df_highqual[['time_utc','qa_val_mean', 'qa_val_mode']].head()  #first row has 0.4 qa_val for both mode and mean
# modequal_ZeroPtFour_highqual = df['qa_val_mode'].iloc[0]
# meanqual_ZeroPtFour_highqual = df['qa_val_mean'].iloc[0]

# +
# print("mode: rows less than 0.4:", df_highqual[['time_utc', 'qa_val_mode']][df_highqual['qa_val_mode'] < modequal_ZeroPtFour_highqual].shape)
# print("mode: rows equal to 0.4:", df_highqual[['time_utc', 'qa_val_mode']][df_highqual['qa_val_mode'] == modequal_ZeroPtFour_highqual].shape)
# print("mode: rows greater than 0.4:", df_highqual[['time_utc', 'qa_val_mode']][df_highqual['qa_val_mode'] > modequal_ZeroPtFour_highqual].shape)
# print("mean: rows less than 0.4:", df_highqual[['time_utc', 'qa_val_mean']][df_highqual['qa_val_mean'] < meanqual_ZeroPtFour_highqual].shape)
# print("mean: rows equal to 0.4:", df_highqual[['time_utc', 'qa_val_mean']][df_highqual['qa_val_mean'] == meanqual_ZeroPtFour_highqual].shape)
# print("mean: rows greater than 0.4:", df_highqual[['time_utc', 'qa_val_mean']][df_highqual['qa_val_mean'] > meanqual_ZeroPtFour_highqual].shape)
# #Due to these distributions of quality numbers, qa_val_mode is ignored because most data is of quality 0.4
# #qa_val_mean however has half the data greater than qa_val >  0.4, therefore use qa_val_mean > 0.4 as higher quality data

# +
# plt.hist(df_highqual['qa_val_mean'], bins=50)

# +
# zones = [x for x in range(1,17)]
# rmse_list = []

# for zone in zones:
#     df_qual = df_highqual[df_highqual['BZone'] == zone ]
#     df_qual = df_qual[df_qual['qa_val_mean']>.4]


#     #Average all methane readings for one day
#     averaged_series = df_qual.groupby('date_formatted')['methane_mixing_ratio_mean'].mean()
#     averaged_df = averaged_series.to_frame().reset_index()

#     #Create new dataframe with the columns named as ds and y, as that is what Prophet requires
#     methane_df = averaged_df.reset_index()[['date_formatted', 'methane_mixing_ratio_mean']].rename({'date_formatted':'ds', 
#                                                                'methane_mixing_ratio_mean':'y'}, 
#                                                               axis='columns')
#     #Create Model and Forecast
#     model, forecast, results, anomaly_df, train, test, rmse, scaler = FBpropet(methane_df, zone=1, changepoint_range=0.95, future_periods=273, anomaly_factor=1)
# #     print("Zone_{} RMSE: {}".format(zone, rmse))
#     rmse_list.append(rmse)

#     #Plot
#     plot_title = 'Zone_{}_Methane_HighQual2'.format(zone)
#     FBpropet_plot(scaler, model, forecast, results, anomaly_df, test, plot_title)
    
# for i in zones:
#     print("Zone_{} RMSE:  {}".format(i, rmse_list[i-1]))
# -


