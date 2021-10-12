# This script installs our python packages on a SageMaker Studio Kernel Application
#!/bin/bash

set -eux

pip install --upgrade pandas numpy geopandas altair geojson matplotlib plotly descartes tensorflow keras seaborn xarray jupytext