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

# # ML Pipeline Template

# This notebook is to be referenced as a template for the ML pipelines we build for the autoencoder framework.

# ## Developer Name: --Insert Name Here--
# ## Date: --Insert Date Here--

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
# from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# -

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 18

# Important to ensure we are using Tensorflow version 2.3
print(tf.__version__)

# ## Paths

ROOT_DIR = '/root/methane/models'
bucket = 'methane-capstone'
subfolder = 'data/data-variants'
s3_path = bucket+'/'+subfolder
print('s3_path: ', s3_path)

# ## Load Data

file_name='data_10D_rn.parquet.gzip' # Insert specific data variant file name here
data_location = 's3://{}/{}'.format(s3_path, file_name)
df = pd.read_parquet(data_location)
print(df.shape)
df.head()

# ## Train / Test Split

# +
# Enter a cutoff date
cutoff = '2021-01-01' # Using 2021 as test data

train = df[df.time_utc < cutoff]
test = df[df.time_utc >= cutoff]
# -

print(train.shape)
train.head()

print(test.shape)
test.head()

# ## Pre-Modeling Work

# Below code cells in this section describe examples of pre-modeling work to be done.

scaler = StandardScaler()
scaler = scaler.fit(train[['methane_mixing_ratio_bias_corrected_mean']])
train['methane_mixing_ratio_bias_corrected_mean'] = scaler.transform(train[['methane_mixing_ratio_bias_corrected_mean']])
test['methane_mixing_ratio_bias_corrected_mean'] = scaler.transform(test[['methane_mixing_ratio_bias_corrected_mean']])

# +
"""
Create windows for LSTM
    - As required by LSTM networks, we require to reshape an input data into 'n_samples' x 'timesteps' x 'n_features'.
    - Number of time steps to look back. Larger sequences (look further back) may improve forecasting.
"""
seq_size = 7

def create_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)


trainX, trainY = create_sequences(train[['methane_mixing_ratio_bias_corrected_mean']], train['methane_mixing_ratio_bias_corrected_mean'], seq_size)
testX, testY = create_sequences(test[['methane_mixing_ratio_bias_corrected_mean']], test['methane_mixing_ratio_bias_corrected_mean'], seq_size)


print("trainX Shape: ", trainX.shape, "trainY Shape: ", trainY.shape)
print("testX Shape: ", testX.shape, "testY Shape: ", testY.shape)
# -

# ## Spatial Embedding Section
#
# A section to work through building out the spatial embeddings representation to be inputted as a layer for the model.



# ## Define Model Config

# Define model config (Add all necessary model arguments and hyperparameters)
config = {
    'units': 64,
    'd_rate': 0.2,
    'loss': 'mae',
    'optimizer': 'adam',
    'epochs': 2,
    'batch_size': 32,
    'val_split': 0.1
}

# ## Define Model Name

# +
dev_name = '' # Put your name here
name_list = file_name.split('.')
freq_res = name_list[0].split('_')
freq = freq_res[1]

if (len(freq_res) == 4):
    resolution = freq_res[2] + freq_res[3]
else:
    resolution = freq_res[2]
    
model_name = f'autoencoder_{dev_name}_{freq}_{resolution}'
print("Model Name: ", model_name)


# -

# ## Create Model

def create_model(config):
    # Initialize the model
    model = keras.Sequential()
    
    # Get model config params
    units = config['units']
    rate = config['d_rate']

    # Create the layers of the model
    model.add(keras.layers.LSTM(units=units, input_shape = (trainX.shape[1], trainX.shape[2])))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.RepeatVector(n=trainX.shape[1]))
    model.add(keras.layers.LSTM(units=units, return_sequences=True))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense( units=trainX.shape[2])))
    
    return model


# Create the model and display its summary
model = create_model(config)
model.summary()


# ## Train Model

def train_model(config):
    # Create the model
    model = create_model(config)
    
    # Get model config values
    loss = config['loss']
    optimizer = config['optimizer']
    epochs = config['epochs']
    batch_size = config['batch_size']
    val_split = config['val_split']
    
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    
    # Set up Tensorboard callback
    log_dir = os.path.join(f"{ROOT_DIR}/logs/{dev_name}", model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(
        trainX, trainY,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split,
        shuffle = False,
        callbacks=[tensorboard_callback]
    )
    
    return model


# +
model = train_model(config)

# View training and validation loss
training_loss = model.history.history['loss']
print("Training Loss: ", training_loss)

validation_loss = model.history.history['val_loss']
print("Validation Loss: ", validation_loss)
# -

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
units = []
d_rate = []
loss = []
optimizer = []
epochs = []
batch_size = []
val_split = []

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

# +


plt.plot(
    test[seq_size:].index,
    scaler.inverse_transform(test[seq_size:].mmrbc_i),
    label = 'methane_mixing_ratio_bias_corrected_mean'
)

sns.scatterplot(
    anomalies.index,
    scaler.inverse_transform(anomalies.mmrbc_i),
    color = sns.color_palette()[3],
    s=60,
    label='anomaly'
)
plt.xticks(rotation=25)
plt.legend()
