import numpy as np
import pandas as pd
# from vectorizer import Vectorizer
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
# from keras.layers import Flatten
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# #from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
# from numpy.random import randint
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
# import joblib

from source.utils import read_xml_qanda, evaluate, generate_data

answer_col = 'answer'
predictor_col = 'correct'
filenames = {'em': 'data/sciEntsBank/EM-post-35.xml',
             'me': 'data/sciEntsBank/ME-inv1-28b.xml',
             'mx': 'data/sciEntsBank/MX-inv1-22a.xml',
             'ps': 'data/sciEntsBank/PS-inv3-51a.xml'}


seed(72) # Python
set_random_seed(72) # Tensorflo

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

decoded_answers = []
for answer in data_answers:
    arr = []
    for i, word in enumerate(answer):
        if i > 0 and word == 0:
            pass
        else:
            arr.append(from_num_dict[word])
    decoded_answers.append(arr)

print(decoded_answers)

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

model = Sequential()
model.add(Embedding(len(from_num_dict), 30, input_length=max_length))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False, input_shape=(max_length,)))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(X_train, y_train, epochs=200, verbose=0)
# evaluate the model
print(f"test_answers shape: {X_train.shape}")
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print('Loss: %f' % loss)

scores = evaluate(model, X_test, y_test)

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
