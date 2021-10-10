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

# +
# # !conda install -c plotly plotly==3.10.0 --yes
# # !conda install -c conda-forge fbprophet --yes
# -

# # Several Options for Modeling Setinel 5p Data with FB Prophet
#
# Model 1 = Entire California  
# Model 2 = Grouped by Latitudes   
# Model 3 = Random Latitude and Longitude Locations, rounded to whole numbers (e.g. Lat = 33 / Long = -131)

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
# import plotly.express as px

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid'] = False

# +
# Read in Data
bucket = 'methane-capstone'
subfolder = 'combined-raw-data'
s3_path_month = bucket+'/'+subfolder
file_name='combined-raw.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)

#Read
df = pd.read_parquet(data_location)

#Convert date into yyyy-mm-dd format
df['date_formatted'] = pd.to_datetime(df['time_utc'], format='%y-%m-%d h%:m%:s%').dt.strftime('%Y-%m-%d')
#Convert to correct Type
df['date_formatted'] = pd.to_datetime(df['date_formatted'])

#Simple EDA
print("SHape: ", df.shape)
print("Types: ", df.dtypes)
df.head()


# -

# # Create FB Prophet Modeling and Graph Functions

def FBpropet(full_df, changepoint_range=0.95, future_periods=273, anomaly_factor=1):

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
    test = full_df[(full_df['ds']>'2020-12-31')]
    
    #Modeling
    model = Prophet(changepoint_range=changepoint_range)   #Create Prophet Model. Default is 80% changepoint
    model.fit(train)                          #Fit on Training     

    #Forecasting
    future = model.make_future_dataframe(periods=future_periods)  #Create Future dataframe for predicting historical and future dates
    forecast = model.predict(future)          #Forecast dataset

    #Create a new dataframe that has the forecasts plus the actual true value for that day
    results = pd.concat([full_df.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower', 'yhat_upper']]],axis=1) 

    #OUTLIER DETECTION
    results['error'] = results['y'] - results['yhat']                      #error = true minus predict
    results['uncertainty'] = results['yhat_upper'] - results['yhat_lower'] #unvertainty = range of prediction interval

    #Look for outliers
    #Create anomaly column that will flag the row as an anomaly
    #If greater than ANOMALY_FACTOR times uncertainty value then it's an outlier
    results['anomaly'] = results.apply(lambda x: 'Yes' if(np.abs(x['error'])>anomaly_factor*x['uncertainty']) else 'No',axis=1)
    results = results.reset_index()
    anomaly_df = results[results.anomaly == 'Yes']
    
    return model, forecast, results, anomaly_df, train, test



def FBpropet_plot(model, forecast, results, anomaly_df, test, plot_title):
    
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

    #Plotting
    fig1 = model.plot(forecast)
    fig1 = plt.scatter(x=test.ds, y=test.y, c='b', marker = ".")
    fig1 = plt.legend(['Train Actuals', 'Forecast', '95% CI', 'Test Actuals'], fontsize=12, loc='upper left')
    fig1 = plt.xlabel("Date", size=18)
    fig1 = plt.ylabel("Methane Concentration (ppb)", size=18)
    fig1 = plt.xticks(fontsize=12)
    fig1 = plt.yticks(fontsize=12)
    fig1 = plt.title(plot_title, size=25)
    fig1 = plt.scatter(x=anomaly_df.ds, y=anomaly_df.y, c='r', s=50)
    fig1 = plt.axvline(datetime(2019, 1, 1))
    fig1 = plt.axvline(datetime(2020, 1, 1))
    fig1 = plt.axvline(datetime(2021, 1, 1))
    #Dark blue are yhat
    #Light blue is yhat confidence interval

    #plot the components
    #Top is the trend
    #Middle is weekly trend
    #Bottom is trend throughout the year
#     fig2 = model.plot_components(forecast)


# # Model_1: Averaged Entire California

# +
#Only quality >0.4
df_qual = df[df['qa_val']>.4]

#Average all methane readings for one day
averaged_series = df_qual.groupby('date_formatted')['methane_mixing_ratio_bias_corrected'].mean()
averaged_df = averaged_series.to_frame().reset_index()

#Create new dataframe with the columns named as ds and y, as that is what Prophet requires
methane_df = averaged_df.reset_index()[['date_formatted', 'methane_mixing_ratio_bias_corrected']].rename({'date_formatted':'ds', 
                                                           'methane_mixing_ratio_bias_corrected':'y'}, 
                                                          axis='columns')
#Create Model and Forecast
model, forecast, results, anomaly_df, train, test = FBpropet(methane_df, changepoint_range=0.95, future_periods=273, anomaly_factor=1)
# -

plot_title = 'Model_1: California Average Methane Concentration'
FBpropet_plot(model, forecast, results, anomaly_df, test, plot_title)

# # Model_2: Partitioned by Latitudes

# +
#Only quality >0.4
df_qual = df[df['qa_val']>.4]

#Average all methane readings for each latitude
lat_averaged_series = df_qual.groupby(['rn_lat','date_formatted'])['methane_mixing_ratio_bias_corrected'].mean()
lat_averaged_df = lat_averaged_series.to_frame().reset_index()
print(lat_averaged_df.rn_lat.unique())
lat_averaged_df.head()

# +
#Separate Each Dataframe By Latitude
lat_list = lat_averaged_df.rn_lat.unique()
lat_df = {}  #dictionary of key=latitude, value = dataframe

for lat in lat_list:
    lat_df[str(lat)[0:2]] = lat_averaged_df[lat_averaged_df.rn_lat==lat]
lat_df['33'].head()

# +
#Create Models for each latitude
lat_models = {}  #dictionary of key=latitude, value = fbprophet's model, forecast, results, anomaly, train, and test data

for lat in lat_df.keys():

    #Create new dataframe with the columns named as ds and y, as that is what Prophet requires
    methane_df = lat_df[lat].reset_index()[['date_formatted', 'methane_mixing_ratio_bias_corrected']].rename({'date_formatted':'ds', 
                                                               'methane_mixing_ratio_bias_corrected':'y'}, 
                                                              axis='columns')
    #Create Model and Forecast
    model, forecast, results, anomaly_df, train, test = FBpropet(methane_df, changepoint_range=0.95, future_periods=273, anomaly_factor=1)
    lat_models[lat] = [model, forecast, results, anomaly_df, train, test]

# +
#Plot for each latitude

for lat in list(lat_models.keys()):

    model = lat_models[lat][0]
    forecast = lat_models[lat][1] 
    results = lat_models[lat][2]
    anomaly_df = lat_models[lat][3]
    train = lat_models[lat][4]
    test = lat_models[lat][5]
    
    plot_title = 'Model_2: California Latitude {} Average Methane Concentration'.format(lat)
    FBpropet_plot(model, forecast, results, anomaly_df, test, plot_title)
# -

# # Model_3: Partitioned by Random Regions

# +
#Create FB Prophet Models With Random Lat/Lon 

#Only quality >0.4
df_qual = df[df['qa_val']>.4]

random_latlon_models = {}
num_models = 5

for i in range(num_models):
    
    #Group dataframe to find all lat and lon combinations (only 1 of each lat/lon combination exist)
    lat_lon_combinations = df_qual.groupby(['rn_lat','rn_lon'])['qa_val'].count().reset_index()

    # Randomly sample latitude and Longitude to run fb prophet
    random=lat_lon_combinations.sample()
    lat_random = float(random.rn_lat)
    lon_random = float(random.rn_lon)
    print("Lat/Lon Random: {:0.1f},{:0.1f}".format(lat_random,lon_random))

    #Subset dataset to only the random lat/lon
    lat_lon_df = df_qual[(df_qual.rn_lat==lat_random) & (df_qual.rn_lon==lon_random)]
    print("Shape: ", lat_lon_df.shape)
    #lat_lon_df.head()
    
    #Average all methane readings for specific lat/lon
    lat_lon_averaged_series = lat_lon_df.groupby(['date_formatted'])['methane_mixing_ratio_bias_corrected'].mean()
    lat_lon_df = lat_lon_averaged_series.to_frame().reset_index()
    methane_df = lat_lon_df.reset_index()[['date_formatted', 'methane_mixing_ratio_bias_corrected']].rename({'date_formatted':'ds', 
                                                                   'methane_mixing_ratio_bias_corrected':'y'}, 
                                                                  axis='columns')

    #Create Model and Forecast
    model, forecast, results, anomaly_df, train, test = FBpropet(methane_df, changepoint_range=0.95, future_periods=273, anomaly_factor=1)
    key = str(lat_random)+str(lon_random)
    random_latlon_models[key] = [model, forecast, results, anomaly_df, train, test]

# +
#Plot for each random lat/lon

for key in list(random_latlon_models.keys()):

    model = random_latlon_models[key][0]
    forecast = random_latlon_models[key][1] 
    results = random_latlon_models[key][2]
    anomaly_df = random_latlon_models[key][3]
    train = random_latlon_models[key][4]
    test = random_latlon_models[key][5]
    lat= key[0:4]
    lon= key[4:]
    
    #Plotting
    plot_title = 'Model_3: CA Lat/Lon ({},{}) Average Methane Concentration'.format(lat,lon)
    FBpropet_plot(model, forecast, results, anomaly_df, test, plot_title)
# -


