from constants import TEAM_NAMES


TEAM_HIST = {}
# 그 전 10경기에 대한 stat

# # Open, High, Low, Volume, Close
# xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
# xy = xy[::-1]  # reverse order (chronically ordered)
# xy = MinMaxScaler(xy)
# x = xy
# y = xy[:, [-1]]  # Close as label

# # build a dataset
# dataX = []
# dataY = []
# for i in range(0, len(y) - seq_length):
#     _x = x[i:i + seq_length]
#     _y = y[i + seq_length]  # Next close price
#     # print(_x, "->", _y)
#     dataX.append(_x)
#     dataY.append(_y)

# # train/test split
# train_size = int(len(dataY) * 0.7)
# test_size = len(dataY) - train_size
# trainX, testX = np.array(dataX[0:train_size]), np.array(
#     dataX[train_size:len(dataX)])
# trainY, testY = np.array(dataY[0:train_size]), np.array(
#     dataY[train_size:len(dataY)])

def create_data(data, train_prop):
	dataX = []
	dataY = []

	for i in range(len(data)):
		# Put home team data
		x_val, y_val = format(data[i])
		dataX.append(x_val)
		dataY.append(y_val)

	train_size = int(len(dataY) * train_prop)
	test_size = len(dataY) - train_size
	trainX, trainY = np.array(dataX[:train_size]), np.array(dataY[:train_size])
	testX, testY = np.array(dataX[train_size:]), np.array(dataY[train_size:])



"""
Formats the data into x_val and y_val
@param JSON value
@returns x_val, y_val
"""
def format(data):
	home = []
	away = []
	# winner = [0] # Winner : 1 if home 0 if away
	result = [
		data['score_board']['summary']['home']['r']
		# data['score_board']['summary']['away']['r']
	] # Home team's score

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

	return home + away, result


