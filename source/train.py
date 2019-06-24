from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import argparse
import os
import pandas as pd

import pickle
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Embedding, LSTM, Dropout, Dense
# from tensorflow.python.training.adam import AdamOptimizer
# tf.enable_eager_execution()

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    pkl_filename = os.path.join(args.model_dir,"model.pkl")
    with open(pkl_filename, 'rb') as file:  
        model = pickle.load(file)

    # load using joblib
    #model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")

    return model


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
    parser.add_argument('--embedding_size', type=int, default=30)
    parser.add_argument('--lstm_size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default='adam')


    # args holds all passed-in arguments
    args = parser.parse_args()

    print(f"epochs={args.epochs} embedding_size={args.embedding_size} lstm_size={args.lstm_size}")

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    test_data = pd.read_csv(os.path.join(training_dir, "test.csv"), header=None, names=None)
    vocab = pd.read_csv(os.path.join(training_dir, "vocab.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]
    max_length = train_x.values.shape[1]

    # Build Model
    model = Sequential()
    model.add(Embedding(len(vocab), args.embedding_size, input_length=max_length))
    model.add(Dropout(args.dropout))
    model.add(LSTM(args.lstm_size, return_sequences=False, input_shape=(max_length,)))
    model.add(Dropout(args.dropout))
    model.add(Dense(1, activation="linear"))
    # optimizer = AdamOptimizer()
    # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['acc'])
    model.compile(optimizer=args.optimizer, loss='mean_squared_error', metrics=['acc'])


    # Train the model
    model.fit(train_x, train_y, epochs=args.epochs, verbose=0)
    
    # Validate
    test_y = test_data.iloc[:, 0]
    test_x = test_data.iloc[:, 1:]
    score = model.evaluate(test_x, test_y, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])

    ## --- End of your code  --- ##

    # Save the trained model
    # Save to file in the current working directory
    pkl_filename = os.path.join(os.environ['SM_MODEL_DIR'],"model.pkl")
    print("Saving model to: {}".format(pkl_filename))
    with open(pkl_filename, 'wb+') as file:
        pickle.dump(model, file)
        
    print("File saved: {}".format(os.path.exists(pkl_filename)))
    # joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
