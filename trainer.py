import os
import sys
import argparse
import json
import tensorflow as tf
import numpy as np

from builder import SeLuModel
from builder import Runner
from constants import *
from formatter import Formatter

DIRNAME = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='KBO Score Prediction Trainer')

parser.add_argument('year', type=int, help='The year of data to train the model with')
parser.add_argument('train_size', type=float, help='The proportion of the training set to the test set')
parser.add_argument('model_name', type=str, help='The name of the model')
parser.add_argument('learn_rate', type=float, help='The learning rate')
parser.add_argument('sequence_length', type=int, help='Sequence length') #TODO
parser.add_argument('epoch', type=int, help='Training epoch')
parser.add_argument('drop_rate', type=float, help='Drop rate')

if __name__ == '__main__':
    args = parser.parse_args()

    # ====== Open data file ======
    file_name = ''
    if args.year == 2017:
        file_name = DATA_17
    elif args.year == 2016:
        file_name = DATA_16
    f = open(DIRNAME + "/" + file_name, 'r')
    print("Loading JSON data")
    data = json.load(f)

    print("Preprocessing the data")
    # formatter class that contains trainX, trainY, testX, testY for individual teams
    formatter = Formatter(data, args.train_size, args.sequence_length)
    trainX_home, trainX_away, trainY, testX_home, testX_away, testY = formatter.get_data()

    ## ======== Build model ======
    with tf.Session() as sess:
        kbo_pred_model = SeLuModel(
            sess, 
            args.model_name, 
            learn_rate=args.learn_rate,
            sequence_length=args.sequence_length
        )
        

        ## ======== Train model ======
        print("Started the training...")
        kbo_runner = Runner()
        kbo_runner.train_run(
            kbo_pred_model, 
            trainX_home, 
            trainX_away, 
            trainY,
            training_epoch=args.epoch, 
            keep_prob=(1 - args.drop_rate)
        )
        print("Training done.")
        print("Start Testing")
        ## ======== Run test =========
        accuracy = kbo_runner.get_accuracy(
            kbo_pred_model, 
            testX_home, 
            testX_away, 
            testY
        )

        print("The percentage of the games predicted correctly")
        print(accuracy)

        ## ======= Save the trained model ======
        print("Saving the trained model...")
        kbo_pred_model.save()
        print("Save complete.")
