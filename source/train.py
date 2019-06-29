from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import argparse
import os
import sys
import pandas as pd

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Dense, Flatten, Reshape
from tensorflow.python.training.adam import AdamOptimizer

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--embedding_size', type=int, default=50)
    parser.add_argument('--flatten', type=int, default=0)
    parser.add_argument('--lstm_dim_1', type=int, default=100)
    parser.add_argument('--lstm_dim_2', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)


    # args holds all passed-in arguments
    args = parser.parse_args()

    print(f"epochs={args.epochs} embedding_size={args.embedding_size} lstm_dim_1={args.lstm_dim_1} lstm_dim_2={args.lstm_dim_2} dropout={args.dropout}")

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    test_data = pd.read_csv(os.path.join(training_dir, "test.csv"), header=None, names=None)
    vocab = pd.read_csv(os.path.join(training_dir, "vocab.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]
    max_answer_len = train_x.values.shape[1]

    # Build Model
    model = Sequential()
    model.add(Embedding(len(vocab), args.embedding_size, input_length=max_answer_len))
    model.add(Dropout(args.dropout))
    if args.flatten:
        model.add(Flatten())
        model.add(Reshape((1, args.embedding_size * max_answer_len)))
    if args.lstm_dim_2:
        model.add(LSTM(args.lstm_dim_1, return_sequences=True))
        model.add(LSTM(args.lstm_dim_2, return_sequences=False))
    else:
        model.add(LSTM(args.lstm_dim_1, return_sequences=False))
    model.add(Dropout(args.dropout))
    model.add(Dense(1, activation="linear"))
    optimizer = AdamOptimizer()
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['acc'])


    # Train the model
    model.fit(train_x, train_y, epochs=args.epochs, verbose=0)
    
    # Validate
    test_y = test_data.iloc[:, 0]
    test_x = test_data.iloc[:, 1:]
    score = model.evaluate(test_x, test_y, verbose=0)
    print(f"Validation_loss:{score[0]};Validation_accuracy:{score[1]};")

    ## --- End of your code  --- ##

    # Save the trained model
    result = tf.contrib.saved_model.save_keras_model(model, os.environ['SM_MODEL_DIR'])
    print(f"Save Keras Model result: {result}")
 