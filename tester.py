"""
https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

RESTful frontend will look something like the above.
"""
import os
import argparse
import tensorflow as tf

DIRNAME = os.path.dirname(os.path.realpath(__file__)) + '/saved_graphs/'



parser = argparse.ArgumentParser(description='KBO Score Prediction Tester')

parser.add_argument('model_name', type=str, help='The trained model to use for prediction')
parser.add_argument('home_team', type=int, help='The home team id')
parser.add_argument('away_team', type=int, help='The away team id')

if __name__ == '__main__':
	args = parser.parse_args()

	with tf.Session() as sess:
		#Call the model.
		saver = tf.train.import_meta_graph(DIRNAME + args.model_name + '.meta')
		# To initialize values with saved data
		saver.restore(sess, args.model_name)
		#Access the graph.
		graph = tf.get_default_graph()

	   
	    # print(sess.run(global_step_tensor)) # returns 1000

		all_vars = tf.get_collection('vars')
		for v in all_vars:
			v_ = sess.run(v)
			print(v_)

