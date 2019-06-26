import pandas as pd


# from keras.layers.core import Dropout
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM


from numpy.random import seed
from tensorflow import set_random_seed

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, LSTM
from tensorflow.python.training.adam import AdamOptimizer

from tensorflow.contrib.saved_model import save_keras_model

from sklearn.model_selection import train_test_split

from source.utils import read_xml_qanda, evaluate, generate_data, decode_answers

answer_col = 'answer'
predictor_col = 'correct'
filenames = {'em': 'data/sciEntsBank/EM-post-35.xml',
             'me': 'data/sciEntsBank/ME-inv1-28b.xml',
             'mx': 'data/sciEntsBank/MX-inv1-22a.xml',
             'ps': 'data/sciEntsBank/PS-inv3-51a.xml'}


seed(72) # Python
set_random_seed(72) # Tensorflow

# question, reference_answer, answer_df = read_xml_qanda(filenames['ps'])
answer_df = pd.DataFrame()
for filename in filenames.values():
    answers = read_xml_qanda(filename)
    answer_df = pd.concat([answer_df, answers], axis=0, ignore_index=True)

print(answer_df)
answer_df.to_csv('data/seb/raw_data.csv', index=False)

data_answers, data_labels, max_length, from_num_dict, to_num_dict = generate_data(answer_df)

print(data_answers)
print(f"Sample Size: {len(data_answers)}")
print(data_labels)
print(f"Longest answer: {max_length}")

# Save Vocabulary
pd.DataFrame(from_num_dict.values()).to_csv('data/seb/vocab.csv', index=False, header=None)

X_train, X_test, y_train, y_test = train_test_split(data_answers, data_labels, test_size=0.30, random_state=72)

# Generate a train.csv file for Sagemaker use
labels_df = pd.DataFrame(y_train)
answers_df = pd.DataFrame(X_train)
test_data = pd.concat([labels_df, answers_df], axis=1)
test_data.to_csv('data/seb/train.csv', index=False)
train_x = test_data.iloc[:, 1:]
max_length = train_x.values.shape[1]
print(f"Test Data columns: {max_length}")


# Generate a test.csv file for predictions
labels_df = pd.DataFrame(y_test)
answers_df = pd.DataFrame(X_test)
test_data = pd.concat([labels_df, answers_df], axis=1)
test_data.to_csv('data/seb/test.csv', index=False)

raw_answers = decode_answers(test_data.values, from_num_dict)
for answer in raw_answers:
    print(f"{answer[0]}. correct: {answer[1]} answer: {' '.join(answer[2:])}")


model = Sequential()
model.add(Embedding(len(from_num_dict), 54, input_length=max_length))
model.add(Dropout(0.22))
model.add(LSTM(100, return_sequences=False, input_shape=(max_length,)))
model.add(Dropout(0.22))
model.add(Dense(1, activation="linear"))
# compile the model
optimizer = AdamOptimizer()
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['acc'])
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test))

# evaluate the model
print(f"test_answers shape: {X_train.shape}")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % loss)

scores = evaluate(model, X_test, y_test)
result = save_keras_model(model, 'model')

print(f"Save Keras Model result: {result}")

# Results
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 72.7%
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

# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 76.2%
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 47, 30)            7020
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 47, 30)            0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               52400
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 59,521
# Trainable params: 59,521
# Non-trainable params: 0

# Accuracy: 97.959184
# Loss: 0.020474

# predictions  0.0  1.0
# actuals
# 0.0           22    3
# 1.0            7   10
#
# Recall:     0.588
# Precision:  0.769
# Accuracy:   0.762

# _________________________________________________________________
# Rerun - Note lower train results in better test accuracy. Possible
# overfitting issue with training.
# _________________________________________________________________
#
# test_answers shape: (98, 47)
# Accuracy: 93.877551
# Loss: 0.055575
# predictions  0.0  1.0
# actuals
# 0.0           20    5
# 1.0            5   12
#
# Recall:     0.706
# Precision:  0.706
# Accuracy:   0.762


# _________________________________________________________________
# Rerun - Reduce Epochs to see if reducing trianing helps...
#
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 100, Train Tst Split: .3 => Accuracy: 73.8%
# _________________________________________________________________
# Accuracy: 89.795918
# Loss: 0.088208
# predictions  0.0  1.0
# actuals
# 0.0           22    3
# 1.0            8    9
#
# Recall:     0.529
# Precision:  0.750
# Accuracy:   0.738

# _________________________________________________________________
# Rerun - Reduce Epochs to see if reducing trianing helps...
#
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 150, Train Tst Split: .3 => Accuracy: 73.8%
# _________________________________________________________________
# test_answers shape: (98, 47)
# Accuracy: 91.836735
# Loss: 0.077527
# predictions  0.0  1.0
# actuals
# 0.0           23    2
# 1.0            9    8
#
# Recall:     0.471
# Precision:  0.800
# Accuracy:   0.738


# _________________________________________________________________
# Rerun - Original
#
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 66.7%
# _________________________________________________________________
# Accuracy: 85.714286
# Loss: 0.123627
# predictions  0.0  1.0
# actuals
# 0.0           24    1
# 1.0           13    4
#
# Recall:     0.235
# Precision:  0.800
# Accuracy:   0.667


# _________________________________________________________________
# Rerun - Increase LSTM nodes
#
# Embedding: 30, Dropout: 0.2, LSTM: 200, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 66.7%
# _________________________________________________________________

# Accuracy: 88.775510
# Loss: 0.096415
# predictions  0.0  1.0
# actuals
# 0.0           23    2
# 1.0           12    5
#
# Recall:     0.294
# Precision:  0.714
# Accuracy:   0.667

# _________________________________________________________________
# Rerun - Decrease LSTM nodes
#
# Embedding: 30, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 71.4%
# _________________________________________________________________
# Accuracy: 100.000000
# Loss: 0.000954

# predictions  -0.0   1.0
# actuals
# 0.0            22     3
# 1.0             9     8
#
# Recall:     0.471
# Precision:  0.727
# Accuracy:   0.714