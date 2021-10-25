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

# # ML Pipeline Template

# This notebook is to be referenced as a template for the ML pipelines we build for the autoencoder framework.

# ## Developer Name: Alyssa
# ## Date: --10/24/21--
#
# ### This notebook tries to implement a LSTM-DNN model similar to the one described in this paper: https://project.inria.fr/aaltd19/files/2019/08/AALTD_19_Karadayi.pdf

# ## Imports

# The following libraries have already been installed due to the AWS Lifecycle Configuration:
# - pandas 
# - numpy 
# - geopandas 
# - altair 
# - geojson 
# - matplotlib 
# - plotly 
# - descartes 
# - tensorflow 
# - keras 
# - tensorboard 
# - seaborn 
# - xarray 
# - jupytext
#
# Add all necessary imports here.

# +
# Downgrade Tensorflow to version 2.3
# # !pip install tensorflow==2.3

# +
import os
import boto3
import datetime
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import Sequential
# from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# -

# Important to ensure we are using Tensorflow version 2.3
print(tf.__version__)

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 18

# ## Paths

ROOT_DIR = '/root/methane/models'
bucket = 'methane-capstone'
subfolder = 'data/data-variants'
s3_path = bucket+'/'+subfolder
print('s3_path: ', s3_path)

# ## Load Data

file_name='data_1D_rn.parquet.gzip' # Insert specific data variant file name here
data_location = 's3://{}/{}'.format(s3_path, file_name)
df = pd.read_parquet(data_location)
print(df.shape)
df.head()

#Create a region field that is a string of the lat and long combined
df['region'] = df['rn_lat'].astype(str) + "," + df['rn_lon'].astype(str)

#Keep only methane column
columns_to_keep = ['time_utc', 'region', 'methane_mixing_ratio_bias_corrected_mean']
df = df[columns_to_keep]
df.head()

#Number of readings per region
count_df = df.groupby(['region']).size().reset_index().rename(columns={0:'count'})
count_df.sort_values('count')

#Turn each region into a column
df = df.pivot(index='time_utc', columns='region', values='methane_mixing_ratio_bias_corrected_mean')\
             .reset_index()
df

#replace nans with 0 for now because it didn't work with NaNs
df = df.fillna(0)

#Format time
df['time_utc'] =  pd.to_datetime(df['time_utc'], format='%Y-%m-%d')
df

# # Break data into timesteps

#Let's focus on the 2 regions with the most data
columns_to_keep = ['time_utc','33.0,-116.0', '34.0,-117.0']
df = df[columns_to_keep]
df

#Only do first 6 months of 2019
df = df.loc[df['time_utc'] > '2018-12-31']
df = df.loc[df['time_utc'] < '2019-07-01']
df

# # Split data into train and test

train_df = df.loc[df['time_utc'] < '2019-06-01']
test_df = df.loc[df['time_utc'] > '2019-05-31']

test_df

# # Format train & test data

# +
#region1 = '33.0,-116.0'
#region2 = '34.0,-117.0'

# +
#Take all methane readings for each region and turn into a list
#train
region1_vals_list_train = train_df['33.0,-116.0'].values.tolist()
region2_vals_list_train = train_df['34.0,-117.0'].values.tolist()

#For each region, create a list of 7 day increments
region1_vals_list_test = test_df['33.0,-116.0'].values.tolist()
region2_vals_list_test = test_df['34.0,-117.0'].values.tolist()


# -

#Function that will create chunks of data in the specified timesteps
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]


# +
#For each region, create a list of 7 day increments
#train
region1_chunked_train = list(divide_chunks(region1_vals_list_train, 7))
region2_chunked_train = list(divide_chunks(region2_vals_list_train, 7))

#test
region1_chunked_test = list(divide_chunks(region1_vals_list_test, 7))
region2_chunked_test = list(divide_chunks(region2_vals_list_test, 7))
# -

# #Remove last list in test data which is 2 days
region1_chunked_test = region1_chunked_test[0:len(region1_chunked_test)-1]
region2_chunked_test = region2_chunked_test[0:len(region2_chunked_test)-1]

# +
print(len(region1_chunked_train))
print(len(region1_chunked_train[0]))
#We are working with 21 weeks for train data

print(len(region1_chunked_test))
print(len(region1_chunked_test[0]))
#We are working with 4 weeks for test data
# -

#Convert to arrays
#Need to make an array for each week
def create_arrays(chunked_region_list):
    final_array = np.empty(7, dtype=object)

    for week in chunked_region_list:
        week_array = np.array(week)
        final_array = np.vstack((final_array, week_array))
        #Remove first empty row
    return final_array[1:]


# +
#Create an array for each region
region1_array_train = create_arrays(region1_chunked_train)
region2_array_train = create_arrays(region2_chunked_train)

region1_array_test = create_arrays(region1_chunked_test)
region2_array_test = create_arrays(region2_chunked_test)

# +
#Combine the arrays for each region
combined_train = np.vstack((region1_array_train, region2_array_train))

combined_test = np.vstack((region1_array_test, region2_array_test))
# -

combined_train.shape
#Indices 0 - 20 are for region 1
#Indices 21 - 42 are for region 2
#Index 0 is for region 1 week 1
#Index 1 is for region 1 week 2, etc.

combined_train[21]

region2_array_train[0]

# +
#Reshape the array 
#.reshape(regions*weeks, 7 (for the timestep we used), 1 feature)
sequence_train = combined_train.reshape((42, 7, 1))
print(sequence_train.shape)

sequence_test = combined_test.reshape((8, 7, 1))
print(sequence_test.shape)
# -

sequence_train[0]

# +
#Standardize

scalers = {}
for i in range(sequence_train.shape[1]):
    scalers[i] = StandardScaler()
    sequence_train[:, i, :] = scalers[i].fit_transform(sequence_train[:, i, :]) 

for i in range(sequence_test.shape[1]):
    sequence_test[:, i, :] = scalers[i].transform(sequence_test[:, i, :]) 
# -

sequence_train[0]

train_final = sequence_train.astype('float32')
test_final = sequence_test.astype('float32')

# # Model

# define model
model = Sequential()
model.add(keras.layers.LSTM(100, activation='relu', input_shape=(7,1)))
model.add(keras.layers.RepeatVector(7))
model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(train_final, train_final, epochs=300, verbose=0)
model.summary()

# demonstrate recreation
yhat = model.predict(train_final, verbose=0)
print(yhat[0,:,0])
print(train_final[0])

# +
# connect the encoder LSTM as the output layer
model2 = Model(inputs=model.inputs, outputs=model.layers[0].output)

# get the feature vector for the input sequence
train_X = model2.predict(train_final)
test_X = model2.predict(test_final)
print(train_X.shape)
print(train_X[1])


# -

def create_labels(final_length):
    #Final length is regions*weeks
    final_array = np.empty(2, dtype=object)

    for i in range(final_length):
        if i < final_length/2:
            new_array = np.array([1,0])
        else:
            new_array = np.array([0,1])
        final_array = np.vstack((final_array, new_array))
        i += 1
    return final_array[1:]


train_Y = create_labels(42)
test_Y = create_labels(8)

dnn_model = Sequential()
dnn_model.add(keras.layers.Dense(200, input_dim=100, activation='relu'))
dnn_model.add(keras.layers.Dense(50, activation='relu'))
dnn_model.add(keras.layers.Dense(2, activation='softmax'))

dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_X_final = train_X.astype('float32')
train_Y_final = train_Y.astype('float32')
test_X_final = test_X.astype('float32')
test_Y_final = test_Y.astype('float32')

dnn_model.fit(train_X_final, train_Y_final, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = dnn_model.evaluate(train_X_final, train_Y_final)
print('Accuracy: %.2f' % (accuracy*100))

#Evaluate model on test data
dnn_model.evaluate(test_X_final, test_Y_final)

#Predict on test data
testX_pred = dnn_model.predict(test_X_final)

testX_pred

#Convert probabilities into 0 and 1
results = []
for item in testX_pred:
    item_results = []
    if item[0] < item[1]:
        item_results.append([0,1])
    else:
        item_results.append([1,0])
    results.append(item_results)

results

# # Alyssa's work ends here

# ## Launch TensorBoard

# Please open up a system terminal and run the following commands:
# > cd ~/methane/models
#
# > pip install tensorboard
#
# > tensorboard --logdir logs/your-name. Replace `your-name` with developer name.
#
# > Visit the URL: https://d-kdgirgbbdmbt.studio.us-west-2.sagemaker.aws/jupyter/default/proxy/insert-port-number-here/. Replace `insert-port-number-here` with the port number shown in terminal output.
#
# > Ctrl C on terminal to exit

# ## Hyperparameter Tuning

# +
# Add more values to arrays to use different hyperparameters and add other parameter variables as necessary
units = [64]
d_rate = [0.2]
loss = ['mae']
optimizer = ['adam']
epochs = [2]
batch_size = [32]
val_split = [0.1]

# Create combinations of all hyperparameters
hyperparams = list(product(units, d_rate, loss, optimizer, epochs, batch_size, val_split))
print(hyperparams)
print(len(hyperparams))

# +
# Write hyperparameters to a dataframe (Add key-value pair for every hyperparameter you want to include in DF)
hp = {
    'units': [i[0] for i in hyperparams],
    'd_rate': [i[1] for i in hyperparams],
    'loss': [i[2] for i in hyperparams],
    'optimizer': [i[3] for i in hyperparams],
    'epochs': [i[4] for i in hyperparams],
    'batch_size': [i[5] for i in hyperparams],
    'val_split': [i[6] for i in hyperparams]
}

hp_df = pd.DataFrame(hp, columns = list(hp.keys()))
hp_df
# -

# #### Get Config For Each Model

configs = hp_df.to_dict(orient='records')
configs

# #### Create And Train Models Using Defined Hyperparameters

# +
trained_models = []
count = 0

for config in configs:
    count += 1
    print(f'Model {count}: {list(config.items())}')
    model = train_model(config)
    trained_models.append((config, model, model.history))
    print()
# -

# #### Launch TensorBoard
#
# Follow the steps described above to re-launch tensorboard to view model performance.

# +
# Add training results into new columns in hyperparameter dataframe
training_loss = []
val_loss = []
training_mae = []
val_mae = []

# Loop through the model histories
for hist in trained_models:
    num_epochs = hist[0]['epochs']
    training_loss.append(hist[2].history['loss'][num_epochs - 1])
    training_mae.append(hist[2].history['mae'][num_epochs - 1])
    val_loss.append(hist[2].history['val_loss'][num_epochs - 1])
    val_mae.append(hist[2].history['mae'][num_epochs - 1])
    
# Add new loss columns to hyperparameters dataframe (results table)
hp_df['training_loss'] = training_loss
hp_df['validation_loss'] = val_loss
hp_df['training_mae'] = training_mae
hp_df['validation_mae'] = val_mae

hp_df
# -

# Write hyperparameters dataframe to csv file
cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
hp_df.to_csv('/root/methane/models/autoencoder/' + dev_name + "/" + model_name + cur_date + '.csv', index = False, header = True)

# ## Save Best Model

# Find model with lowest validation mae to save the best model
best_model_index = hp_df[hp_df.validation_mae == hp_df.validation_mae.min()].index
best_model = trained_models[best_model_index[0]][1]
best_model_name = f'{model_name}_{cur_date}'
best_model.save(f'/root/methane/models/autoencoder/{dev_name}/{best_model_name}.h5')

# ## Upload Best Model Artifact To S3

client = boto3.client('s3')
client.upload_file(Filename=f'/root/methane/models/autoencoder/{dev_name}/{best_model_name}.h5', Bucket=bucket, Key=f'models/autoencoder/{best_model_name}.h5')

# ## Plot Best Model

# +
# plot_model(best_model, to_file = f'./{model_name}/images/model_{}_{}.png', show_shapes=True, show_layer_names=True)
# -

# ## Evaluate Best Model On Test Data

best_model.evaluate(testX, testY)

# ## Predict On Test Data

testX_pred = best_model.predict(testX)
test_mae_loss = np.mean(np.abs(testX_pred, testX), axis=1)

# ## Anomaly Detection

# +
"""
Selecting the threshold is key!
    - We can examine the training losses and let that guide our choice for the threshold
    - If we lower the threshold value, more anomalies will be detected
    - We can reason about this using descriptive statistics we have learned from the data
"""

anom_thresh = 1.5

test_score_df = pd.DataFrame(index=test[seq_size:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = anom_thresh
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['methane_mixing_ratio_bias_corrected_mean'] = test[seq_size:].methane_mixing_ratio_mean


# Plot the Test Loss against Anomaly Loss Threshold
plt.plot(test_score_df.index, test_score_df.loss, label = 'loss')
plt.plot(test_score_df.index, test_score_df.threshold, label = 'threshold')
plt.xticks(rotation=25)
plt.title("Test Loss vs. Anomaly Loss Threshold")
plt.legend()

anomalies = test_score_df[test_score_df.anomaly]
print("Anomalies DF Shape: ", anomalies.shape)
anomalies
# -



# +
plt.plot(
    test[seq_size:].index,
    scaler.inverse_transform(test[seq_size:].methane_mixing_ratio_bias_corrected_mean),
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.methane_mixing_ratio_bias_corrected_mean),
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
# -


