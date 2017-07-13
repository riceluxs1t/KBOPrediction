"""
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

RESTful frontend will look something like the above.
"""
import os
import argparse
import tensorflow as tf

from builder import SeLuModel

DIRNAME = os.path.dirname(os.path.realpath(__file__)) + '/saved_graphs/'


parser = argparse.ArgumentParser(description='KBO Score Prediction Tester')

parser.add_argument('model_name', type=str, help='The pretrained model to use')
parser.add_argument('home_team', type=int, help='The home team id')
parser.add_argument('away_team', type=int, help='The away team id')

if __name__ == '__main__':
	args = parser.parse_args()

	with tf.Session() as sess:
		kbo_pred_model = SeLuModel(
			sess, 
			args.model_name, 
			learn_rate=0
		)

		# Call data
		# home = get_team(args.home_team)
		# away = get_team(args.home_team)
		# Concat home & away
		# home_r, away_r = kbo_pred_model.predict([home + away])

		print("Prediction :")
		print("Home :", home_r)
		print("Away :", away_r)
