# Methane Anomaly Detector Pipeline

Here we provide the notebooks used to extract raw data and process into a model-readable format to detect anomalies using an Autoencoder.  
These notebooks are tested on the Amazon's Sagemaker environment and all data is stored in an AWS S3 bucket.  


## Notebooks  

### data_pipeline.py: Extracts raw data and process into a model-readable format.
Raw data includes methane data from Sentinel 5P and weather data from ERA5, both accessible through Amazon's Registry of Open Data on AWS.  
Data processing includes merging methane, weather, and CA climate zone datasets, data grouping, and merging new data to old dataset.

### multi_model_loadpretrainedmodels.py: Pretrains 16 autoencoder models for anomaly detection.  
Reads processed datasets as inputs to an Long Short-Term Memory (LSTM) Autoencoder model.  
Data processing includes standardizing data, interpolating missing data, and generating the data input format into Kera's Tensorflow package.  
Results are stored in .h5 format for the LSTM Autoencoder models and .pickle format for the fitted StandardScaler model (for standardizing data).  

### multi_model_pretraining.py: Predicts/Infers anomalies on a new dataset by utilizing pretrained models.  
Reads processed dataset as inputs to a pretrained LSTM Autoencoder Model.
Predicts/Infers anomalies on the input data and generates helpful visuals.  
