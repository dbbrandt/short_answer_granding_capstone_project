import pandas as pd


# from keras.layers.core import Dropout
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM


from numpy.random import seed
from tensorflow import set_random_seed

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, LSTM, Input, Masking
from tensorflow.python.training.adam import AdamOptimizer

from tensorflow.contrib.saved_model import save_keras_model

from sklearn.model_selection import train_test_split

from source.utils import read_xml_qanda, evaluate, generate_data, decode_answers

answer_col = 'answer'
predictor_col = 'correct'

def sag_data():
    # filename = 'data/sag2/answers.csv'
    filename = 'data/sag2/balanced_answers.csv'

    seed(72) # Python
    set_random_seed(72) # Tensorflow

    answer_df = pd.read_csv(filename, dtype={'id': str})

    print(answer_df)

    data_answers, data_labels, max_length, from_num_dict, to_num_dict = generate_data(answer_df, 0.1, 'id')

    print(data_answers)
    print(f"Sample Size: {len(data_answers)}")
    print(data_labels)
    print(f"Longest answer: {max_length}")

    # Save Vocabulary
    pd.DataFrame(from_num_dict.values()).to_csv('data/sag2/vocab.csv', index=False, header=None)

    X_train, X_test, y_train, y_test = train_test_split(data_answers, data_labels, test_size=0.30, random_state=72)

    # Generate a train.csv file for Sagemaker use
    labels_df = pd.DataFrame(y_train)
    answers_df = pd.DataFrame(X_train)
    test_data = pd.concat([labels_df, answers_df], axis=1)
    test_data.to_csv('data/sag2/train.csv', index=False)
    train_x = test_data.iloc[:, 1:]
    max_length = train_x.values.shape[1]
    print(f"Test Data columns: {max_length}")

    # Generate a test.csv file for predictions
    labels_df = pd.DataFrame(y_test)
    answers_df = pd.DataFrame(X_test)
    test_data = pd.concat([labels_df, answers_df], axis=1)
    test_data.to_csv('data/sag2/test.csv', index=False)

    raw_answers = decode_answers(test_data.values, from_num_dict)
    for answer in raw_answers:
        print(f"{answer[0]}. correct: {answer[1]} answer: {' '.join(answer[2:])}")

    return X_train, y_train, X_test, y_test, max_length, from_num_dict

X_train, y_train, X_test, y_test, max_length, from_num_dict = sag_data()

EPOCHS = 10
# INIT_LR = 1e-3

model = Sequential()
model.add(Input(shape=(max_length,)))
model.add(Masking(mask_value=0., input_shape=(max_length,)))
model.add(Embedding(len(from_num_dict), 10, input_length=max_length))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False, input_shape=(max_length,)))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))
# model.add(Dense(1, activation="sigmoid"))
# compile the model
optimizer = AdamOptimizer()
# optimizer=AdamOptimizer(learning_rate=INIT_LR)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['acc'])
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['acc'])

# summarize the model
print(model.summary())

# fit the model
model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_test, y_test))

# evaluate the model
print(f"test_answers shape: {X_train.shape}")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % loss)

scores = evaluate(model, X_test, y_test)

pred = model.predict(X_test)
print(pred)
# result = save_keras_model(model, 'model')

# print(f"Save Keras Model result: {result}")

# Results
# Embedding: 200, Dropout: 0.2, LSTM: 200, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 72.7%
# Racal: 1   never predics false
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 47, 30)            4050
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 47, 30)            0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               52400
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 56,551
# Trainable params: 56,551
# Non-trainable params: 0

# predictions  0.0  1.0
# actuals
# 0.0           21    4
# 1.0            9    8
#
# Recall:     0.471
# Precision:  0.667
# Accuracy:   0.690

# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 74%  Recal: 1
# Never predics false
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# _________________________________________________________________
# =================================================================
# embedding (Embedding)        (None, 175, 200)          392600
# _________________________________________________________________
# dropout (Dropout)            (None, 175, 200)          0
# _________________________________________________________________
# lstm (LSTM)                  (None, 200)               320800
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 200)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 201
# =================================================================
# Total params: 713,601
# Trainable params: 713,601
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 1709 samples, validate on 733 samples
# WARNING:tensorflow:From /Users/dbrandt/Dropbox/PycharmProjects/short_answer_granding_capstone_project/venv/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# 2019-06-27 14:03:44.621063: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# Epoch 1/10
# 1709/1709 [==============================] - 18s 11ms/sample - loss: 0.2302 - acc: 0.6969 - val_loss: 0.1998 - val_acc: 0.7244
# Epoch 2/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2032 - acc: 0.7262 - val_loss: 0.2005 - val_acc: 0.7244
# Epoch 3/10
# 1709/1709 [==============================] - 18s 10ms/sample - loss: 0.2027 - acc: 0.7262 - val_loss: 0.2020 - val_acc: 0.7244
# Epoch 4/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2016 - acc: 0.7262 - val_loss: 0.2097 - val_acc: 0.7244
# Epoch 5/10
# 1709/1709 [==============================] - 18s 11ms/sample - loss: 0.2063 - acc: 0.7262 - val_loss: 0.2019 - val_acc: 0.7244
# Epoch 6/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2042 - acc: 0.7256 - val_loss: 0.2029 - val_acc: 0.7244
# Epoch 7/10
# 1709/1709 [==============================] - 18s 10ms/sample - loss: 0.2055 - acc: 0.7262 - val_loss: 0.2023 - val_acc: 0.7244
# Epoch 8/10
# 1709/1709 [==============================] - 16s 10ms/sample - loss: 0.2029 - acc: 0.7267 - val_loss: 0.2034 - val_acc: 0.7244
# Epoch 9/10
# 1709/1709 [==============================] - 18s 10ms/sample - loss: 0.2051 - acc: 0.7244 - val_loss: 0.2073 - val_acc: 0.7244
# Epoch 10/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2035 - acc: 0.7262 - val_loss: 0.2006 - val_acc: 0.7244
# test_answers shape: (1709, 175)
# Accuracy: 72.615564
# Loss: 0.199718
# predictions  1.0
# actuals
# 0.0          202
# 1.0          531
#
# Recall:     1.000
# Precision:  0.724
# Accuracy:   0.724

# Embedding: 400, Dropout: 0.2, LSTM: 400, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Balanced data 1342 points 50/50 split correct/incorrect.
# Accuracy: 50.479233
# Loss: 0.261770
# predictions  0.0  1.0
# actuals
# 0.0            1  205
# 1.0            0  197
#
# Recall:     1.000
# Precision:  0.490
# Accuracy:   0.491

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Balanced data 40 points 50/50 split correct/incorrect.
# Loss: 0.483871
# predictions  1.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     1.000
# Precision:  0.585
# Accuracy:   0.585

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Added Learning Rate and Decay to Adam optimizer. learning_rate = 1e-3
# Accuracy: 51.612902
# Loss: 0.250435
# predictions  1.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     1.000
# Precision:  0.585
# Accuracy:   0.585



# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Switch from linear activation to sigmoid, switch from mean_squared_error to binary_crossentropy. This worked best for
# my other answer validation.
# Balanced data 40 points 50/50 split correct/incorrect.
# NOTE: All predictions are still the same at 51.3 which is possibly due to the fact that the padding makes most of the
# input data 0's due to the wide range of input lengths.
# Accuracy: 51.612902
# Loss: 0.692645
# predictions  1.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     1.000
# Precision:  0.585
# Accuracy:   0.585
#
# [[0.51313174]
#  [0.51313174]
#  [0.51313174]

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Switch back to Linear and mean_squared_error and added a Masking layer to mask 0's
# Balanced data 40 points 50/50 split correct/incorrect.
# Just shifted all the answers to false
# Accuracy: 48.387095
# Loss: 0.255791
# /Users/dbrandt/Dropbox/PycharmProjects/short_answer_granding_capstone_project/source/utils.py:61: RuntimeWarning: invalid value encountered in long_scalars
#   precision = tp / (tp + fp)
# predictions  0.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     0.000
# Precision:  nan
# Accuracy:   0.415
#
# [[0.4383411 ]
#  [0.4383411 ]

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Switch back to Linear and mean_squared_error and added a Masking layer to mask 0's
# Balanced data 40 points 50/50 split correct/incorrect.
# Just shifted all the answers to false
