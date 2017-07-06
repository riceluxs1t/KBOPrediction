"""
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

RESTful frontend will look something like the above.
"""
import os
import argparse
import tensorflow as tf

from builder import SeLuModel

DIRNAME = os.path.dirname(os.path.realpath(__file__)) + '/saved_graphs/'

ex = [[1 for i in range(58)]]


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

		y = kbo_pred_model.predict(ex)

		print(y)
