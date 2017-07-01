"""
This is a runner for KBO Prediction.
"""
import json
import os
import sys
import tensorflow as tf
import numpy as np
from model import Model
from model import Runner
from constants import *


DIRNAME = os.path.split(os.path.abspath(sys.argv[0]))[0]

## ====== Create data set from 2017 data =====

"""
Call JSON file
"""
f = open(DIRNAME + "/" + DATA_17, 'r')
print("Load JSON data")
data = json.load(f)

"""
Formats the data into x_val and y_val
@param JSON value
@returns x_val, y_val
"""
def format(data):
	home = []
	away = []
	winner = [0] # Winner : 1 if home 0 if away

	# Get the winner
	if data['score_board']['summary']['home']['r'] > data['score_board']['summary']['away']['r']:
		winner[0] = 1

	for k1, v1 in data['score_board']['summary'].items():
		if k1 == 'home':
			for k2, v2 in v1.items():
				if k2 != 'r':
					home.append(v2)
		else:
			for k2, v2 in v1.items():
				if k2 != 'r':
					away.append(v2)

	for k1, v1 in data['pitcher_info'].items():
		if k1 == 'home':
			for k2, v2 in v1[0].items():
				if k2 != 'name':
					if k2 == 'era':
						home.append(float(v2))
					else:	
						home.append(v2)
		else:
			for k2, v2 in v1[0].items():
				if k2 != 'name':
					if k2 == 'era':
						away.append(float(v2))
					else:	
						away.append(v2)

	for k1, v1 in data['batter_info'].items():
			if k1 == 'home':
				for k2, v2 in v1[0].items():
					if k2 != 'name':
						if k2 == 'hra':
							home.append(float(v2))
						else:
							home.append(v2)
			else:
				for k2, v2 in v1[0].items():
					if k2 != 'name':
						if k2 == 'hra':
							away.append(float(v2))
						else:
							away.append(v2)

	for k1, v1 in data['away_team_standing'].items():
		if k1 != 'name':
			away.append(v1)

	for k1, v1 in data['home_team_standing'].items():
		if k1 != 'name':
			home.append(v1)

	return home + away, winner


# home, away, y = format(data[0])
# print(len(home), len(away))
# print(home)
# print(away)
# print(y)

"""
Create data set and divide into test data set & train data set
"""
dataX = []
dataY = []

for i in range(len(data)):
	# Put home team data
	x_val, y_val = format(data[i])
	dataX.append(x_val)
	dataY.append(y_val)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, trainY = np.array(dataX[:train_size]), np.array(dataY[:train_size])
testX, testY = np.array(dataX[train_size:]), np.array(dataY[train_size:])

## ======== Build model ======
sess = tf.Session()
kbo_pred_model = Model(sess, "Model1")

## ======== Train model ======
kbo_runner = Runner()
kbo_runner.train_run(kbo_pred_model, trainX, trainY)


## ======== Run test =========
accuracy = kbo_runner.get_accuracy(kbo_pred_model, testX, testY)

print("Model Accuracy: ")
print(accuracy)










