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

# ### Goal:
#
# Parse the Google Doc that Jaclyn put together for all scanlines that run through California from `11/28/2018 - 10/01/2021`
# Save the output as a dictionary `{file_name: size_in_MB}`
#
# Google doc downloaded as a `.txt` file. Filename `ca_s5p_20181128_20211002.txt`

# +
import boto3
import pandas as pd
from sagemaker import get_execution_role

import os
import subprocess
import re
import time
import pickle
import json
import math


role = get_execution_role()

# -

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


file_test_string = 'S5P_OFFL_L2__CH4____20210930T183132_20210930T201301_20549_02_020200_20211002T102455'
print(len(file_test_string))

size_test_string = 'Mission: Sentinel-5 P  Instrument: TROPOMI  Sensing Date: 2018-11-29T19:13:31.000Z  Size: 42.26 MB'
print(len(size_test_string))

# ### Get all the filenames

# +
file_name = './data_maps/ca_s5p_20181128_20211002.txt'

#TOTAL count expected to be --> 1849
file_names = []

with open(file_name, mode='r', encoding='utf-8-sig') as file:
    for ind, line in enumerate(file, 0):
        line = line.strip()
        #Using length to easily grab all the filenames
        if len(line) == 83:
            fn = line + '.nc'
            cur_file_name = fn
            file_names.append(cur_file_name)

    
# -

len(file_names)

# ### Get the total data size

# +
file_sizes = []
with open(file_name, mode='r', encoding='utf-8-sig') as file:
    for ind, line in enumerate(file, 0):
        line = line.strip()

        #Using length to easily grab all the filenames
        if "Size:" in line:
            size = line.split(":")[-1].strip()
            file_sizes.append(size)
            

# -

len(file_sizes)

all_ca_files = dict(zip(file_names, file_sizes))

size_total = 0.0
for k,v in all_ca_files.items():
    size_total += float(v.split(" ")[0])


size_total

# ### Total Data: `90-92gb`

with open('./data_maps/ca_filenames.pickle', 'wb') as handle:
    pickle.dump(all_ca_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
