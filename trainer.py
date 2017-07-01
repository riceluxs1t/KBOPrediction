"""
This is a runner for KBO Prediction.
"""
import os
import sys
import argparse
import json
import tensorflow as tf
import numpy as np

from model import SeLuModel
from model import ReLuModel
from model import Runner
from constants import *


DIRNAME = os.path.split(os.path.abspath(sys.argv[0]))[0]

"""
Formats the data into x_val and y_val
@param JSON value
@returns x_val, y_val
"""
def format(data):
	home = []
	away = []
	# winner = [0] # Winner : 1 if home 0 if away
	winner = [data['score_board']['summary']['home']['r']] # Home team's score

	# Get the winner
	# if data['score_board']['summary']['home']['r'] > data['score_board']['summary']['away']['r']:
	# 	winner[0] = 1

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


parser = argparse.ArgumentParser(description='KBO Score Prediction SELU NN Trainer')

parser.add_argument('file_name', type=str, help='The data file (must be in the same directory')
parser.add_argument('model_name', type=str, help='The name of the model')
parser.add_argument('learn_rate', type=float, help='The learning rate')
parser.add_argument('epoch', type=int, help='Training epoch')
parser.add_argument('drop_rate', type=float, help='Drop rate')
parser.add_argument('-s', '--selu', action="store_true", help='Detemine whether to use SeLU')

if __name__ == '__main__':
	args = parser.parse_args()
	"""
	Create data set and divide into test data set & train data set
	"""
	f = open(DIRNAME + "/" + args.file_name, 'r')
	print("Load JSON data")
	data = json.load(f)

	dataX = []
	dataY = []

	for i in range(len(data)):
		# Put home team data
		x_val, y_val = format(data[i])
		dataX.append(x_val)
		dataY.append(y_val)

	# train_size = int(len(dataY) * 0.7)
	# test_size = len(dataY) - train_size
	# trainX, trainY = np.array(dataX[:train_size]), np.array(dataY[:train_size])
	# testX, testY = np.array(dataX[train_size:]), np.array(dataY[train_size:])

	## ======== Build model ======
	with tf.Session() as sess:
		kbo_pred_model = None
		if args.selu:
			kbo_pred_model = SeLuModel(
				sess, 
				args.model_name, 
				learn_rate=args.learn_rate
			)
		else:
			kbo_pred_model = ReLuModel(
				sess, 
				args.model_name, 
				learn_rate=args.learn_rate
			)

		## ======== Train model ======
		kbo_runner = Runner()
		# kbo_runner.train_run(kbo_pred_model, trainX, trainY, training_epoch=2000, keep_prob=0.7)
		kbo_runner.train_run(
			kbo_pred_model, 
			dataX, 
			dataY,
			training_epoch=args.epoch, 
			keep_prob=(1 - args.drop_rate)
		)

		## ======= Save the trained model ======
		saver = tf.train.Saver()
		saver.save(sess, DIRNAME + '/' + args.model_name + 'graph.chkp')








