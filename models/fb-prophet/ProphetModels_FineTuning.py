# -*- coding: utf-8 -*-
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

# # This notebook fine-tunes a single model.
#
# Model 1 = Entire California  
#
# https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning  
#
# TLDR:  
# ***changepoint_prior_scale***: Default = 0.05,  flexibility of the trend. small = underfit  
# ***seasonality_prior_scale***: Default = 10 ,   flexibility of the seasonality.  
# ***changepoint_range***:       Default = 0.8,   proportion of the history in which trend is allowed to change.   (over/underfit)  
#
# **Parameters that can be tuned**:   
#
# ***changepoint_prior_scale***: This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.
#
# ***seasonality_prior_scale***: This parameter controls the flexibility of the seasonality. Similarly, a large value allows the seasonality to fit large fluctuations, a small value shrinks the magnitude of the seasonality. The default is 10., which applies basically no regularization. That is because we very rarely see overfitting here (there’s inherent regularization with the fact that it is being modeled with a truncated Fourier series, so it’s essentially low-pass filtered). A reasonable range for tuning it would probably be [0.01, 10]; when set to 0.01 you should find that the magnitude of seasonality is forced to be very small. This likely also makes sense on a log scale, since it is effectively an L2 penalty like in ridge regression.
#
# ***holidays_prior_scale***: This controls flexibility to fit holiday effects. Similar to seasonality_prior_scale, it defaults to 10.0 which applies basically no regularization, since we usually have multiple observations of holidays and can do a good job of estimating their effects. This could also be tuned on a range of [0.01, 10] as with seasonality_prior_scale.
#
# ***seasonality_mode***: Options are ['additive', 'multiplicative']. Default is 'additive', but many business time series will have multiplicative seasonality. This is best identified just from looking at the time series and seeing if the magnitude of seasonal fluctuations grows with the magnitude of the time series (see the documentation here on multiplicative seasonality), but when that isn’t possible, it could be tuned.
#
# **Maybe tune?**
#
# ***changepoint_range***: This is the proportion of the history in which the trend is allowed to change. This defaults to 0.8, 80% of the history, meaning the model will not fit any trend changes in the last 20% of the time series. This is fairly conservative, to avoid overfitting to trend changes at the very end of the time series where there isn’t enough runway left to fit it well. With a human in the loop, this is something that can be identified pretty easily visually: one can pretty clearly see if the forecast is doing a bad job in the last 20%. In a fully-automated setting, it may be beneficial to be less conservative. It likely will not be possible to tune this parameter effectively with cross validation over cutoffs as described above. The ability of the model to generalize from a trend change in the last 10% of the time series will be hard to learn from looking at earlier cutoffs that may not have trend changes in the last 10%. So, this parameter is probably better not tuned, except perhaps over a large number of time series. In that setting, [0.8, 0.95] may be a reasonable range.  
#
# **Parameters that would likely not be tuned**
#
# ***growth***: Options are ‘linear’ and ‘logistic’. This likely will not be tuned; if there is a known saturating point and growth towards that point it will be included and the logistic trend will be used, otherwise it will be linear.
#
# ***changepoints***: This is for manually specifying the locations of changepoints. None by default, which automatically places them.
#
# ***n_changepoints***: This is the number of automatically placed changepoints. The default of 25 should be plenty to capture the trend changes in a typical time series (at least the type that Prophet would work well on anyway). Rather than increasing or decreasing the number of changepoints, it will likely be more effective to focus on increasing or decreasing the flexibility at those trend changes, which is done with changepoint_prior_scale.
#
# ***yearly_seasonality***: By default (‘auto’) this will turn yearly seasonality on if there is a year of data, and off otherwise. Options are [‘auto’, True, False]. If there is more than a year of data, rather than trying to turn this off during HPO, it will likely be more effective to leave it on and turn down seasonal effects by tuning seasonality_prior_scale.
#
# ***weekly_seasonality***: Same as for yearly_seasonality.
#
# ***daily_seasonality***: Same as for yearly_seasonality.
#
# ***holidays***: This is to pass in a dataframe of specified holidays. The holiday effects would be tuned with holidays_prior_scale.
#
# ***mcmc_samples***: Whether or not MCMC is used will likely be determined by factors like the length of the time series and the importance of parameter uncertainty (these considerations are described in the documentation).
#
# ***interval_width***: Prophet predict returns uncertainty intervals for each component, like yhat_lower and yhat_upper for the forecast yhat. These are computed as quantiles of the posterior predictive distribution, and interval_width specifies which quantiles to use. The default of 0.8 provides an 80% prediction interval. You could change that to 0.95 if you wanted a 95% interval. This will affect only the uncertainty interval, and will not change the forecast yhat at all and so does not need to be tuned.
#
# ***uncertainty_samples***: The uncertainty intervals are computed as quantiles from the posterior predictive interval, and the posterior predictive interval is estimated with Monte Carlo sampling. This parameter is the number of samples to use (defaults to 1000). The running time for predict will be linear in this number. Making it smaller will increase the variance (Monte Carlo error) of the uncertainty interval, and making it larger will reduce that variance. So, if the uncertainty estimates seem jagged this could be increased to further smooth them out, but it likely will not need to be changed. As with interval_width, this parameter only affects the uncertainty intervals and changing it will not affect in any way the forecast yhat; it does not need to be tuned.
#
# ***stan_backend***: If both pystan and cmdstanpy backends set up, the backend can be specified. The predictions will be the same, this will not be tuned.

# # Import Packages and Data

# +
#Packages 
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

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

def FBpropet(full_df, changepoint_range=0.95, changepoint_prior_scale=0.05, 
             seasonality_mode='additive', seasonality_prior_scale=10, future_periods=273, 
             anomaly_factor=1):

    """
    Inputs:  
        "full_df" formatted with two columns, 'ds' = datetime, 'y' = value
        
        "changepoint_range" for % of data, 0.95 will place potential chagnepoints 
        in the first 95% of the time series. higher changepoint_range = can overfit to train data
        
        "changepoint_prior_scale": Default = 0.05, flexibility of the trend.  
        
        "seasonality_prior_scale": Default = 10 , flexibility of the seasonality.

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
    model = Prophet(changepoint_range=changepoint_range, 
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode)             #Create Prophet Model. Default is 80% changepoint
    model.fit(train)                                               #Fit on Training     

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
# WITHOUT TUNING

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
model, forecast, results, anomaly_df, train, test = FBpropet(methane_df, changepoint_range=0.95, 
                                                             changepoint_prior_scale=0.05,
                                                             seasonality_prior_scale=10, 
                                                             future_periods=273, 
                                                             anomaly_factor=1)
#Plotting
plot_title = 'Model_1: California Average Methane Concentration'
FBpropet_plot(model, forecast, results, anomaly_df, test, plot_title)
# -

# # Hyper-Parameter Tuning

# +
#TUNING 
from fbprophet.diagnostics import cross_validation, performance_metrics
import time

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode': ['additive', 'multiplicative']
}


# param_grid = {  
#     'changepoint_prior_scale': [0.001],
#     'seasonality_prior_scale': [0.01],
#     'seasonality_mode': ['additive', 'multiplicative']
# }

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

start1=time.time()
start2=time.time()

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(train)  # Fit model with given params
    df_cv = cross_validation(m, horizon='30 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])
    
    
    future_periods=273
    anomaly_factor=1
    full_df = train
    
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

    plot_title = 'CA Avg Methane, CP/S/S-mode={}/{}/{}'.format(params['changepoint_prior_scale'], params['seasonality_prior_scale'], params['seasonality_mode'])

    FBpropet_plot(m, forecast, results, anomaly_df, test, plot_title)
    
    print("time in sec: ",time.time()-start1)
    start1=time.time()
    
    
# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

best_params = all_params[np.argmin(rmses)]
print('best param: ', best_params)
print("time to complete: ", time.time()-start2, "seconds")
# -










