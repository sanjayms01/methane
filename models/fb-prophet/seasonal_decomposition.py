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

# # Imports

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings
import copy
from pylab import rcParams

# # Read in full data from s3

# +
# Read in Total Dataframe
bucket = 'methane-capstone/data'
subfolder = 'combined-raw-data'
s3_path_month = bucket+'/'+subfolder

#Read in parquet file
file_name='combined-raw.parquet'
data_location = 's3://{}/{}'.format(s3_path_month, file_name)

#Read
df = pd.read_parquet(data_location)
# -

# # Format data

# +
#Convert date into yyyy-mm-dd format
df['date_formatted'] = pd.to_datetime(df['time_utc'], format='%y-%m-%d h%:m%:s%').dt.strftime('%Y-%m-%d')

#Convert date to date timestamp format (otherwise pandas thinks it's a string)
df['date_formatted'] = pd.to_datetime(df['date_formatted'])

#Average all methane readings for one day
averaged_series = df.groupby('date_formatted')['methane_mixing_ratio_bias_corrected'].mean()
averaged_df = averaged_series.to_frame()

averaged_df
# -

# # Show seasonal decomposition

# Additive Model
# * y(t) = Trend + Seasonality + Noise
#
# Multiplicative Model
# * y(t) = Trend * Seasonality * Noise
#

# +
#Look at each of the values individually

#result.trend
#result.seasonal
result.resid
# result.observed

# +
#Model can be additive or multiplicative
#A rule of thumb for selecting the right model is to see in our plot if the trend and seasonal variation are relatively constant over time, in other words, linear. 
#If yes, then we will select the Additive model. 
#Otherwise, if the trend and seasonal variation increase or decrease over time then we use the Multiplicative model.

result = seasonal_decompose(averaged_df['methane_mixing_ratio_bias_corrected'], model='multiplicative', period=365, extrapolate_trend = 'freq')
# -

result.seasonal.plot()

result.trend.plot()

result.plot()

# # Remove seasonality and trend

# +
#Remove seasonality
#Divide because we used multiplicative model
# https://www.machinelearningplus.com/time-series/time-series-analysis-python/

deseasonalized = averaged_df.methane_mixing_ratio_bias_corrected / result.seasonal
deseasonalized
# -

#Remove trend
detrended = averaged_df.methane_mixing_ratio_bias_corrected / result.trend
detrended

#Remove seasonality and trend
deseasonalized = averaged_df.methane_mixing_ratio_bias_corrected / result.seasonal
both_removed = deseasonalized / result.trend
both_removed

# # Summarized code to remove seasonality

# +
# Create a dataframe of data. The date should be the index.

# Run seasonal decomposition
# Can set model to 'multiplicative' or 'additive'
# result = seasonal_decompose(averaged_df['methane_mixing_ratio_bias_corrected'], model='multiplicative', period=365, extrapolate_trend = 'freq')

# Remove seasonality
# deseasonalized = averaged_df.methane_mixing_ratio_bias_corrected / result.seasonal

# Can add this new deseasonalized data back to original dataframe and use the deseasonalized numbers as the new outcome variable
# -


