"""
This is a runner for KBO Prediction.
"""
import json
import os
import sys

from model import Model
from model import Runner
from constants import *

DIRNAME = os.path.split(os.path.abspath(sys.argv[0]))[0]

# Create data set from 2017 data
f = open(DIRNAME + "/" + DATA_17, 'r')
print("Load JSON data")
data = json.load(f)

print(len(data))



# Build model


# Train model


# Run test