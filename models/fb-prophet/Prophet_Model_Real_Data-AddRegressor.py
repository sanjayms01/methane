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

# +
# # !pip install pandas --upgrade

# +
# # !pip install numpy --upgrade

# +
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
# -

# ## Combine Monthly Data Parquet Files into 1 Dataframe

# + jupyter={"source_hidden": true}
# #Read in all files
# bucket = 'methane-capstone'
# subfolder = 'month-raw-data'
# file_path = bucket+'/'+subfolder

# data_2018_11 = 's3://{}/{}'.format(file_path, '2018/2018-11-meth.parquet.gzip')
# data_2018_12 = 's3://{}/{}'.format(file_path, '2018/2018-12-meth.parquet.gzip/')

# data_2019_1 = 's3://{}/{}'.format(file_path, '2019/2019-01-meth.parquet.gzip')
# data_2019_2 = 's3://{}/{}'.format(file_path, '2019/2019-02-meth.parquet.gzip')
# data_2019_3 = 's3://{}/{}'.format(file_path, '2019/2019-03-meth.parquet.gzip')
# data_2019_4 = 's3://{}/{}'.format(file_path, '2019/2019-04-meth.parquet.gzip')
# data_2019_5 = 's3://{}/{}'.format(file_path, '2019/2019-05-meth.parquet.gzip')
# data_2019_6 = 's3://{}/{}'.format(file_path, '2019/2019-06-meth.parquet.gzip')
# data_2019_7 = 's3://{}/{}'.format(file_path, '2019/2019-07-meth.parquet.gzip')
# data_2019_8 = 's3://{}/{}'.format(file_path, '2019/2019-08-meth.parquet.gzip')
# data_2019_9 = 's3://{}/{}'.format(file_path, '2019/2019-09-meth.parquet.gzip')
# data_2019_10 = 's3://{}/{}'.format(file_path, '2019/2019-10-meth.parquet.gzip')
# data_2019_11 = 's3://{}/{}'.format(file_path, '2019/2019-11-meth.parquet.gzip')
# data_2019_12 = 's3://{}/{}'.format(file_path, '2019/2019-12-meth.parquet.gzip')

# data_2020_1 = 's3://{}/{}'.format(file_path, '2020/2020-01-meth.parquet.gzip')
# data_2020_2 = 's3://{}/{}'.format(file_path, '2020/2020-02-meth.parquet.gzip')
# data_2020_3 = 's3://{}/{}'.format(file_path, '2020/2020-03-meth.parquet.gzip')
# data_2020_4 = 's3://{}/{}'.format(file_path, '2020/2020-04-meth.parquet.gzip')
# data_2020_5 = 's3://{}/{}'.format(file_path, '2020/2020-05-meth.parquet.gzip')
# data_2020_6 = 's3://{}/{}'.format(file_path, '2020/2020-06-meth.parquet.gzip')
# data_2020_7 = 's3://{}/{}'.format(file_path, '2020/2020-07-meth.parquet.gzip')
# data_2020_8 = 's3://{}/{}'.format(file_path, '2020/2020-08-meth.parquet.gzip')
# data_2020_9 = 's3://{}/{}'.format(file_path, '2020/2020-09-meth.parquet.gzip')
# data_2020_10 = 's3://{}/{}'.format(file_path, '2020/2020-10-meth.parquet.gzip')
# data_2020_11 = 's3://{}/{}'.format(file_path, '2020/2020-11-meth.parquet.gzip')
# data_2020_12 = 's3://{}/{}'.format(file_path, '2020/2020-12-meth.parquet.gzip')

# data_2021_1 = 's3://{}/{}'.format(file_path, '2021/2021-01-meth.parquet.gzip')
# data_2021_2 = 's3://{}/{}'.format(file_path, '2021/2021-02-meth.parquet.gzip')
# data_2021_3 = 's3://{}/{}'.format(file_path, '2021/2021-03-meth.parquet.gzip')
# data_2021_4 = 's3://{}/{}'.format(file_path, '2021/2021-04-meth.parquet.gzip')
# data_2021_5 = 's3://{}/{}'.format(file_path, '2021/2021-05-meth.parquet.gzip')
# data_2021_6 = 's3://{}/{}'.format(file_path, '2021/2021-06-meth.parquet.gzip')
# data_2021_7 = 's3://{}/{}'.format(file_path, '2021/2021-07-meth.parquet.gzip')
# data_2021_8 = 's3://{}/{}'.format(file_path, '2021/2021-08-meth.parquet.gzip')
# data_2021_9 = 's3://{}/{}'.format(file_path, '2021/2021-09-meth.parquet.gzip')



# +
# #Combine all csv files into 1 dataframe
# df = pd.concat(map(pd.read_parquet, [data_2018_11, data_2018_12,
# data_2019_1,
# data_2019_2 ,
# data_2019_3 ,
# data_2019_4 ,
# data_2019_5 ,
# data_2019_6 ,
# data_2019_7 ,
# data_2019_8 ,
# data_2019_9 ,
# data_2019_10,
# data_2019_11,
# data_2019_12,

# data_2020_1,
# data_2020_2 ,
# data_2020_3,
# data_2020_4 ,
# data_2020_5 ,
# data_2020_6 ,
# data_2020_7 ,
# data_2020_8 ,
# data_2020_9 ,
# data_2020_10 ,
# data_2020_11 ,
# data_2020_12 ,

# data_2021_1 ,
# data_2021_2 ,
# data_2021_3 ,
# data_2021_4 ,
# data_2021_5 ,
# data_2021_6 ,
# data_2021_7,
# data_2021_8,
# data_2021_9 ]), ignore_index=True)
# df

# +
# #Write the dataframe to 1 parquet file
# file_name='methane_fake_data_all.parquet'
# df.to_parquet('s3://{}/{}'.format(file_path,file_name))

# +
# Read in Total Dataframe
bucket = 'methane-capstone'
subfolder = 'combined-raw-data'
s3_path_month = bucket+'/'+subfolder

#Read in parquet file
file_name='combined-raw.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)

#Read
df = pd.read_parquet(data_location)
# -

print(df.shape)
df.head()

# ## Facebook Prophet Work

# ### Cleaning and Formatting Data

#Convert date into yyyy-mm-dd format
df['date_formatted'] = pd.to_datetime(df['time_utc'], format='%y-%m-%d h%:m%:s%').dt.strftime('%Y-%m-%d')
df

df = df[df['qa_val']>.4]
df

# ## Model with Latitude as Additional Regressor

#Create dataframe with time, latitude and methane reading
averaged_df = df[['time_utc','lat','methane_mixing_ratio_bias_corrected']].groupby('time_utc').mean().reset_index()
averaged_df

#Check data types
averaged_df.info()

#Plot data
averaged_df.plot(kind='line',x='time_utc',y='methane_mixing_ratio_bias_corrected', title='California Methane Emissions')
plt.show()

#Create new dataframe with the columns named as ds and y, as that is what Prophet requires
methane_df = averaged_df.reset_index()[['time_utc', 'methane_mixing_ratio_bias_corrected']].rename({'time_utc':'ds', 
                                                           'methane_mixing_ratio_bias_corrected':'y'}, 
                                                          axis='columns')
methane_df

#Add latitude
combined_df = averaged_df.reset_index()[['time_utc', 'methane_mixing_ratio_bias_corrected', 'lat']].rename({'time_utc':'ds', 
                                                           'methane_mixing_ratio_bias_corrected':'y', 'lat':'latitude'}, 
                                                          axis='columns')
combined_df

#Split dataframe into train and test
train = combined_df[(methane_df['ds']<'2021-01-01')]
test = combined_df[(methane_df['ds']>'2020-12-31')]


# +
#train = train.sample(1000)

print(train.shape)
print(test.shape)
# -

#Create a prophet model
#Changepoint range is the confidence interval of the output
#Default is 80%
model = Prophet(changepoint_range=0.95)
model.add_regressor('latitude', prior_scale=0.5, mode='multiplicative')

#Fit the model on the training data
model.fit(train)

# +
#Create dataframe that is just date (ds) and latitude
latitude_df = combined_df[['ds', 'latitude']]
latitude_df

#Number of periods should be equal to the length of the test data
#Default frequency is daily
#Can set freq='H' to predict at hourly increments
#future = model.make_future_dataframe(periods=1500)
#future = pd.merge(future, latitude_df, on='ds')
#future = latitude_df.sample(1500)

future = latitude_df
future
#Display. Only shows dates
#len(future.ds.unique())
# -

#Now predict for the future dates
#yhat = actual predicted value
#yhat_lower & yhat_upper = the confidence interval
forecast = model.predict(future)

#Print the forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#Create a new dataframe that has the forecasts plus the actual true value for that day
results = pd.concat([methane_df.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower', 'yhat_upper']]],axis=1)
results

# +
fig1 = model.plot(forecast)
fig1 = plt.scatter(x=test.ds, y=test.y, c='b', marker = ".")
fig1 = plt.legend(['Train Actuals', 'Forecast', '95% CI', 'Test Actuals'])
fig1 = plt.xlabel("Date")
fig1 = plt.ylabel("Methane Concentration (ppb)")
fig1 = plt.title("California Average Methane Concentration")

#Black points are actual
#Dark blue are yhat
#Light blue is yhat confidence interval
# -

comp = model.plot_components(forecast)
#plot the components
#This will take the time series data and show the trend and seasonality trend out of it
#Top is the trend
#Middle is weekly trend
#Bottom is trend throughout the year

results.head()

#Take difference in actual value and predicted value
results['error'] = results['y'] - results['yhat']

#Calculate the confidence interval
results['uncertainty'] = results['yhat_upper'] - results['yhat_lower']

results

# # Model with Fake Avg Temperature as Additional Regressor

#Average all methane readings for one day
averaged_series_weather = df.groupby('date_formatted')['methane_mixing_ratio_bias_corrected'].mean()
averaged_df_weather = averaged_series_weather.to_frame().reset_index()
averaged_df_weather

#Convert date to date timestamp format (otherwise pandas thinks it's a string)
averaged_df_weather['date_formatted'] = pd.to_datetime(averaged_df_weather['date_formatted'])
averaged_df_weather

#Check data types
averaged_df_weather.info()

#Plot data
averaged_df_weather.plot(kind='line',x='date_formatted',y='methane_mixing_ratio_bias_corrected', title='California Methane Emissions')
plt.show()

#Create new dataframe with the columns named as ds and y, as that is what Prophet requires
methane_df_weather= averaged_df_weather.reset_index()[['date_formatted', 'methane_mixing_ratio_bias_corrected']].rename({'date_formatted':'ds', 
                                                           'methane_mixing_ratio_bias_corrected':'y'}, 
                                                          axis='columns')
methane_df_weather

#Read in weather data
weather_df = pd.read_csv('fake_weather_data.csv')
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df = weather_df.rename(columns={'date': 'ds'})
weather_df

#Adds weather
combined_df_weather = pd.merge(methane_df_weather, weather_df, on='ds')
combined_df_weather

#Split dataframe into train and test
train_weather = combined_df_weather[(methane_df_weather['ds']<'2021-01-01')]
test_weather = combined_df_weather[(methane_df_weather['ds']>'2020-12-31')]

print(train_weather.shape)
print(test_weather.shape)

#Create a prophet model
#Changepoint range is the confidence interval of the output
#Default is 80%
model_weather = Prophet(changepoint_range=0.95)
model_weather.add_regressor('avg_temp', prior_scale=0.5, mode='multiplicative')

#Fit the model on the training data
model_weather.fit(train_weather)

# +
#Number of periods should be equal to the length of the test data
#Default frequency is daily
#Can set freq='H' to predict at hourly increments
future_weather = model_weather.make_future_dataframe(periods=273)
future_weather = pd.merge(future_weather, weather_df, on='ds')

future_weather
#Display. Only shows dates
#len(future.ds.unique())


# +
#Now predict for the future dates
#yhat = actual predicted value
#yhat_lower & yhat_upper = the confidence interval
forecast_weather = model_weather.predict(future_weather)

#Print the forecast
forecast_weather[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# -

#Create a new dataframe that has the forecasts plus the actual true value for that day
results_weather = pd.concat([methane_df_weather.set_index('ds')['y'], forecast_weather.set_index('ds')[['yhat','yhat_lower', 'yhat_upper']]],axis=1)

results_weather

# +
fig1 = model_weather.plot(forecast_weather)
fig1 = plt.scatter(x=test_weather.ds, y=test_weather.y, c='b', marker = ".")
fig1 = plt.legend(['Train Actuals', 'Forecast', '95% CI', 'Test Actuals'])
fig1 = plt.xlabel("Date")
fig1 = plt.ylabel("Methane Concentration (ppb)")
fig1 = plt.title("California Average Methane Concentration")

#Black points are actual
#Dark blue are yhat
#Light blue is yhat confidence interval
# -

comp = model_weather.plot_components(forecast_weather)
#plot the components
#This will take the time series data and show the trend and seasonality trend out of it
#Top is the trend
#Middle is weekly trend
#Bottom is trend throughout the year

results_weather.head()

# ### Begin outlier detection portion

#Take difference in actual value and predicted value
results['error'] = results['y'] - results['yhat']

#Calculate the confidence interval
results['uncertainty'] = results['yhat_upper'] - results['yhat_lower']

results

# +
#Look for outliers
#Take error component and take absolute value
#If greater than 1.5 times uncertainty value then it's an outlier
#Sometimes if don't use 1.5 and just use uncertainty then may predict too many outliers
#results[results['error'].abs()> 1.5*results['uncertainty']]

#Changing to not use 1.5, because shows no outliers
results[results['error'].abs()> 1*results['uncertainty']]
# -

#Create anomaly column that will flag the row as an anomaly
#Make sure to update the value * uncertainty here, to match what is above
results['anomaly'] = results.apply(lambda x: 'Yes' if(np.abs(x['error'])>1*x['uncertainty']) else 'No',axis=1)
results = results.reset_index()

results.tail(10)

results.anomaly.astype('category').cat.codes

sp_names = ['Not Anomaly', 'Anomaly']
scatter = plt.scatter(x=results.ds,
            y=results.y,
            c=results.anomaly.astype('category').cat.codes, cmap='Accent')
plt.legend(handles=scatter.legend_elements()[0], 
           labels=sp_names)
plt.xlabel("Date")
plt.ylabel("Methane Concentration (ppb)")
plt.title("California Average Methane Concentration")

comp = model.plot_components(forecast)

help(Prophet.seasonality_mode)

anomaly_df = results[results.anomaly == 'Yes']
anomaly_df

fig1 = model.plot(forecast)
fig1 = plt.scatter(x=test.ds, y=test.y, c='b', marker = ".")
fig1 = plt.legend(['Train Actuals', 'Forecast', '95% CI', 'Test Actuals'], fontsize=12, loc='upper left')
fig1 = plt.xlabel("Date", size=18)
fig1 = plt.ylabel("Methane Concentration (ppb)", size=18)
fig1 = plt.xticks(fontsize=12)
fig1 = plt.yticks(fontsize=12)
fig1 = plt.title("California Average Methane Concentration", size=25)
fig1 = plt.scatter(x=anomaly_df.ds, y=anomaly_df.y, c='r', s=50)
fig1 = plt.axvline(datetime(2019, 1, 1))
fig1 = plt.axvline(datetime(2020, 1, 1))
fig1 = plt.axvline(datetime(2021, 1, 1))

model.params

methane_df

# +
from fbprophet.diagnostics import cross_validation, performance_metrics

cv_results = cross_validation(model =model, initial = '370 days', horizon = '6 days')
df_p = performance_metrics(cv_results)
df_p
# -

# # Calculate MSE

from sklearn.metrics import mean_absolute_error

results_nonan = results[results.y > 0]
results_nonan.head()

y_true = results_nonan['y'].values
y_pred = results_nonan['yhat'].values
print(len(y_true))
print(len(y_pred))

mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)

# # Change Points

#add changepoints 
#https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet
from fbprophet.plot import add_changepoints_to_plot
fig = model.plot(forecast)
a=add_changepoints_to_plot(fig.gca(),model,forecast)  #just shows the change point lines

# +
# If Over or Underfitting, change "changepoint_prior_scale".  increasing makes trend more flexible

#Create a prophet model
model_overunderfit = Prophet(changepoint_prior_scale=500)
model_overunderfit.fit(train)
forecast = model_overunderfit.predict(future)
fig = model_overunderfit.plot(forecast)
a=add_changepoints_to_plot(fig.gca(),model_overunderfit,forecast)

# +
# If Over or Underfitting, change "changepoint_prior_scale".  increasing makes trend more flexible

#Create a prophet model
model_overunderfit = Prophet(changepoint_prior_scale=0.01)
model_overunderfit.fit(train)
forecast = model_overunderfit.predict(future)
fig = model.plot(forecast)
a=add_changepoints_to_plot(fig.gca(),model,forecast)
# -

# Specify Change Points
#Create a prophet model
model_cpoint = Prophet(changepoints=['2019-08-01'])
model_cpoint.fit(train)
forecast = model_cpoint.predict(future)
fig = model_cpoint.plot(forecast)
a=add_changepoints_to_plot(fig.gca(),model_cpoint,forecast)







