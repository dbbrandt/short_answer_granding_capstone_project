import numpy as np
import pandas as pd
# from vectorizer import Vectorizer
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
# from keras.layers import Flatten
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# #from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
# from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import joblib

from data_preprocess import read_xml_qanda

answer_col = 'answer'
predictor_col = 'correct'
filenames = {'em': 'data/sciEntsBank/EM-post-35.xml',
             'me': 'data/sciEntsBank/ME-inv1-28b.xml',
             'mx': 'data/sciEntsBank/MX-inv1-22a.xml',
             'ps': 'data/sciEntsBank/PS-inv3-51a.xml'}

def validate(predictors, target, model):
    pred = model.predict(predictors).flatten().round()
    score = accuracy_score(target, pred.flatten())
    return score

def encode_answers(answers):
    vocab_string = " ".join(answers)
    vocab_list = vocab_string.split(' ')
    vocabulary = sorted(list(dict.fromkeys(vocab_list)))
    to_num_dict = {word: i for i, word in enumerate(vocabulary)}
    from_num_dict = {i: word for i, word in enumerate(vocabulary)}

    #print(vocabulary)

    encoded_answers = []
    for answer in answers:
        arr = []
        for word in answer.split(' '):
            arr.append(to_num_dict[word])
        encoded_answers.append(arr)

    #print(encoded_answers)
    max_length = max([len(answer) for answer in encoded_answers])
    padded_answers = pad_sequences(encoded_answers, maxlen=max_length, padding='post').tolist()
    padded_answers = np.array(padded_answers).astype('float')
    return max_length, padded_answers, from_num_dict, to_num_dict


def generate_data(filename):
    question, reference_answer, answer_df = read_xml_qanda(filename)

    max_length, encoded_answers, from_num_dict, to_num_dict = encode_answers(answer_df[answer_col].values)

    encoded_answer_df = pd.DataFrame(encoded_answers)
    encoded_answer_df[predictor_col] = answer_df[predictor_col].astype(float)

    # randomize data
    randomized_data = encoded_answer_df.reindex(np.random.permutation(encoded_answer_df.index))
    randomized_labels = randomized_data[predictor_col].values
    randomized_answers = randomized_data.drop([predictor_col], axis=1).values

    return randomized_answers, randomized_labels, max_length, from_num_dict, to_num_dict


test_answers, test_labels, max_length, from_num_dict, to_num_dict = generate_data(filenames['em'])

print(test_answers)
print(test_labels)
print(max_length)

decoded_answers = []
for answer in test_answers:
    arr = []
    for i, word in enumerate(answer):
        if i > 0 and word == 0:
            pass
        else:
            arr.append(from_num_dict[word])
    decoded_answers.append(arr)

print(decoded_answers)

X_train, X_test, y_train, y_test = train_test_split(test_answers, test_labels, test_size=0.20, random_state=53)

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

score = validate(X_test, y_test, model)
print('Prediction Accuracy: %f' % (score*100))

# Results
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .2 => Accuracy: 62.5
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

# Accuracy: 100.000000
# Loss: 0.000692
# Prediction Accuracy: 62.500000