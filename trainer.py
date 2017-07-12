import os
import sys
import argparse
import json
import tensorflow as tf
import numpy as np

from builder import RNN
from builder import Runner
from constants import *
# from formatter import create_data
from formatter import formatter

DIRNAME = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='KBO Score Prediction Trainer')

parser.add_argument('year', type=int, help='The year of data to train the model with')
parser.add_argument('train_size', type=float, help='The proportion of the training set to the test set')
parser.add_argument('learn_rate', type=float, help='The learning rate')
parser.add_argument('epoch', type=int, help='Training epoch')
parser.add_argument('sequence_length', type=float, help='Sequence length')
parser.add_argument('stack_num', type=float, help='The size of LSTM cell stack')


if __name__ == '__main__':
    args = parser.parse_args()

    # ====== Open data file ======
    file_name = ''
    if args.year == 2017:
        file_name = DATA_17
    elif args.year == 2016:
        file_name = DATA_16
    f = open(DIRNAME + "/" + file_name, 'r')
    print("Load JSON data")
    data = json.load(f)

    # formatter class that contains trainX, trainY, testX, testY for individual teams
    team_date = formatter(data, args.train_size, args.sequence_length)

    for team in TEAM_NAMES:
        ## ======== Build model ======
        with tf.Session() as sess:
            kbo_pred_model = RNN(
                sess,
                TEAM_NAMES[team],
                learn_rate=args.learn_rate,
                hidden_size=args.hidden_size,
                sequence_length=args.sequence_length,
                stack_num=args.stack_num
            )

        ## ======== Call team data ======
        trainX, trainY, testX, testY = team_date.get_data(TEAM_NAMES[team])

        ## ======== Train model ======
        print("Started the training the model for ", team)
        kbo_runner = Runner()
        kbo_runner.train_run(
            kbo_pred_model,
            trainX,
            trainY,
            training_epoch=args.epoch
        )
        print("Done training")
        print("Started testing")
        ## ======== Run test =========
        accuracy = kbo_runner.get_accuracy(kbo_pred_model, testX, testY)

        print("Model Average Error: ")
        print(accuracy)
        ## ======= Save the trained model ======
        print("Saving the trained model...")
        kbo_pred_model.save()
        print("Save complete.")
