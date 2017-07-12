from collections import defaultdict
from constants import *
import numpy as np


class formatter:
	"""
	Given the season data, create train data and test data for each team and store it in the dictionary
	s.t has team name as the key and data set as the value.
	"""
	def __init__(self, data, train_size, seq_length):
		self.train_size = train_size
		self.seq_length = seq_length
		self.team_data = defaultdict(list)
		self.team_hist = defaultdict(list)
		self.create_hist(data)
		self.create_data()

	"""
	Put all the records of a team to a dictionary. The result being at the
	last element of the array.
	"""
	def create_hist(self, data):
		for _ in range(0, len(data)):
			home_hist = []
			away_hist = []

			for k1, v1 in data['score_board']['scores'].items():
				if k1 == 'home':
					home_hist += v1
				else:
					away_hist += v1

			for k1, v1 in data['score_board']['summary'].items():
				if k1 == 'home':
					for k2, v2 in v1.items():
						if k2 != 'r':
							home_hist.append(v2)
				else:
					for k2, v2 in v1.items():
						if k2 != 'r':
							away_hist.append(v2)

			for k1, v1 in data['away_team_standing'].items():
				if k1 != 'name':
					away_hist.append(v1)

			for k1, v1 in data['home_team_standing'].items():
				if k1 != 'name':
					home_hist.append(v1)

			home_hist.append(data['score_board']['summary']['home']['r'])
			away_hist.append(data['score_board']['summary']['away']['r'])

			self.team_hist[data['home_team_name']].append(home_hist)
			self.team_hist[data['away_team_name']].append(away_hist)

	"""
	For each team put trainX, trainY, testX, testY in the team_data dictionary.
	"""
	def create_data(self):
		for team in TEAM_NAMES.values():
			xy = np.array(self.team_hist[team])
			x = xy
			y = xy[:, [-1]]

			dataX = []
			dataY = []
			for i in range(0, len(self.team_hist[team]) - self.seq_length):
				dataX.append(x[i : i + self.seq_length])
				dataY.append(y[i + self.seq_length])

			train_size = int(len(dataY) * self.train_size)
			trainX, trainY = np.array(dataX[:train_size]), np.array(dataY[:train_size])
			testX, testY = np.array(dataX[train_size:]), np.array(dataY[train_size:])

			self.team_data[team] = [trainX, trainY, testX, testY]

	"""
	Getter for the team's data
	"""
	def get_data(self, team_name):
		return self.team_data[team_name]