from collections import defaultdict
from constants import *
import numpy as np


class Formatter:
	"""
	Given the season data, create train data and test data for each team and store it in the dictionary
	s.t has team name as the key and data set as the value.
	"""
	def __init__(self, data, train_size, seq_length):
		self.seq_length = seq_length
		self.train_size = train_size
		self.team_hist = defaultdict(list)
		self.create_data(data)

	"""
	Put all the records of a team to a dictionary. The result being at the
	last element of the array.
	"""
	def create_data(self, data):

		dataX_home = []
		dataX_away = []
		dataY = []
		
		for game in data:
			home_hist = []
			away_hist = []

			for k1, v1 in game['score_board']['summary'].items():
				if k1 == 'home':
					for k2, v2 in v1.items():
						if k2 != 'r':
							home_hist.append(v2)
				else:
					for k2, v2 in v1.items():
						if k2 != 'r':
							away_hist.append(v2)

			for k1, v1 in game['away_team_standing'].items():
				if k1 != 'name':
					away_hist.append(v1)

			for k1, v1 in game['home_team_standing'].items():
				if k1 != 'name':
					home_hist.append(v1)

			home_hist.append(game['score_board']['summary']['home']['r'])
			away_hist.append(game['score_board']['summary']['away']['r'])

			if len(self.team_hist[game['home_team_name']]) >= self.seq_length and len(self.team_hist[game['away_team_name']]) >= self.seq_length:
				dataX_home.append(sum(self.team_hist[game['home_team_name']][-self.seq_length:], [])) # Concat all 
				dataX_away.append(sum(self.team_hist[game['away_team_name']][-self.seq_length:], []))
				dataY.append([home_hist[-1], away_hist[-1]])

			self.team_hist[game['home_team_name']].append(home_hist)
			self.team_hist[game['away_team_name']].append(away_hist)

		train_size = int(len(dataY) * self.train_size)
		trainX_home, trainX_away, trainY = np.array(dataX_home[:train_size]), np.array(dataX_away[:train_size]), np.array(dataY[:train_size])
		testX_home, testX_away, testY = np.array(dataX_home[train_size:]), dataX_away[train_size:], np.array(dataY[train_size:])

		self.team_data = [trainX_home, trainX_away, trainY, testX_home, testX_away, testY]

	"""
	Getter for the team's data
	Later to be used to predict the result
	"""
	def get_data(self):
		return self.team_data

