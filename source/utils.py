from xml.dom import minidom
import pandas as pd
import numpy as np
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup
from numpy.random import seed

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from tensorflow import set_random_seed
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, LSTM, Flatten, Reshape
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.contrib.saved_model import save_keras_model, load_keras_model

answer_col = 'answer'
predictor_col = 'correct'
question_id_file = 'data/sag2/files'

def get_xml_elements(doc, tag):
    elements = doc.getElementsByTagName(tag)
    values = [e.firstChild for e in elements]
    return values, elements

def read_xml_qanda(filename):
    doc = minidom.parse(filename)
    values, elements = get_xml_elements(doc, 'questionText')
    question = values[0].data
    values, elements = get_xml_elements(doc, 'referenceAnswer')
    reference_answer = values[0].data
    # print(question)
    # print(reference_answer)
    values, elements = get_xml_elements(doc, 'studentAnswer')
    answers = []
    for e in elements:
        correct = 1 if e.getAttribute('accuracy') == 'correct' else 0
        answers.append([e.firstChild.data, correct])
    answers_df = pd.DataFrame(answers, columns=['answer','correct'])
    answers_df['question'] = question
    answers_df['reference_answer'] = reference_answer
    return answers_df

def load_seb_data():
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

    data_answers, data_labels, max_length, from_num_vocab, to_num_vocab = generate_data(answer_df)

    print(data_answers)
    print(f"Sample Size: {len(data_answers)}")
    print(data_labels)
    print(f"Longest answer: {max_length}")
    print(f"Vocabulary Size:{len(from_num_vocab)}")
    # Save Vocabulary
    pd.DataFrame(from_num_vocab.values()).to_csv('data/seb/vocab.csv', index=False, header=None)

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

    raw_answers = decode_answers(test_data.values, from_num_vocab)
    for answer in raw_answers:
        print(f"{answer[0]}. correct: {answer[1]} answer: {' '.join(answer[2:])}")

    return X_train, y_train, X_test, y_test, max_length, from_num_vocab

def load_sag_data(percent_of_data=1):
    filename = 'data/sag2/answers.csv'
    # filename = 'data/sag2/balanced_answers.csv'

    seed(72) # Python
    set_random_seed(72) # Tensorflow

    answer_df = pd.read_csv(filename, dtype={'id': str})

    print(answer_df)

    data_answers, data_labels, max_length, from_num_dict, to_num_dict = generate_data(answer_df, percent_of_data, 'id')

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
    test_data.to_csv('data/sag2/train.csv', index=False, header=False)
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

def build_model(params):
    model = Sequential()
    model.add(Embedding(params['vocab_size'], params['embedding_dim'], input_length=params['max_answer_len']))
    model.add(Dropout(params['dropout']))
    if params['flatten']:
        model.add(Flatten())
        model.add(Reshape((1, params['embedding_dim'] * params['max_answer_len'])))
    if params['lstm_dim_2']:
        model.add(LSTM(params['lstm_dim_1'], return_sequences=True))
        model.add(LSTM(params['lstm_dim_2'], return_sequences=False))
    else:
        model.add(LSTM(params['lstm_dim_1'], return_sequences=False))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation="linear"))

    # compile the model
    optimizer = AdamOptimizer()
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['acc'])
    print(model.summary())
    return model

def fit_model(model, model_dir, epochs, X_train, y_train):
    # evaluate the model
    print(f"test_answers shape: {X_train.shape}")
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % loss)

    # evaluate the model
    print(f"test_answers shape: {X_train.shape}")
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    print('Loss: %f' % loss)

    # fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose=1) #, validation_data=(X_test, y_test))
    save_keras_model(model, model_dir)

    return model

def load_model(model_dir):
    return load_keras_model(model_dir)


def save_model(model, model_dir):
    save_keras_model(model, model_dir)

def train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test, train=False):

    if train:
        # Build Model
        model = build_model(model_params)
        fit_model(model, model_dir, model_params['epochs'], X_train, y_train)
    else:
        # Load existing model
        filename = f"{model_dir}/{model_file}"
        model = load_model(filename)
        print(model.summary())

    evaluate(model, X_test, y_test)

    print('Predict on all data')
    X_data = np.append(X_train, X_test, axis=0)
    y_data = np.append(y_train, y_test, axis=0)
    evaluate(model, X_data, y_data)

    return model

def evaluate(predictor, test_features, test_labels, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """

    # rounding and squeezing array
    test_preds = np.squeeze(predictor.predict(test_features))
    min_pred = min(test_preds)
    max_pred = max(test_preds)
    print(f"Min preds: {min_pred} max_preds: {max_pred}")
    test_preds = (test_preds - min_pred)/(max_pred - min_pred)
    test_preds = np.round(test_preds)

    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1 - test_labels, test_preds).sum()
    tn = np.logical_and(1 - test_labels, 1 - test_preds).sum()
    fn = np.logical_and(test_labels, 1 - test_preds).sum()

    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # print metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actuals'], colnames=['predictions']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

def simplified_answer(answer):
    # nltk.download("stopwords", quiet=True)

    text = BeautifulSoup(answer, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\-]", " ", text.lower())  # Convert to lower case
    words = text.split()  # Split string into words
    # words = [w for w in words if w not in stopwords.words("english")]  # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words]  # stem

    return ' '.join(words)

def id_map():
    ids = pd.read_csv(question_id_file, dtype={'id': str})['id'].values
    id_to_num = {id: i for i, id in enumerate(ids)}
    return id_to_num

def encode_answers(raw_answers, category_col=None, category_data=[]):
    answers = [simplified_answer(answer) for answer in raw_answers]
    # answers = raw_answers
    vocab_string = " ".join(answers)
    vocab_list = vocab_string.split()
    vocabulary = sorted(list(dict.fromkeys(vocab_list)))
    to_num_dict = {word: i for i, word in enumerate(vocabulary)}
    from_num_dict = {i: word for i, word in enumerate(vocabulary)}

    id_to_num = id_map() if category_col else []

    encoded_answers = []
    for index, answer in enumerate(answers):
        arr = [float(id_to_num[category_data[index]])] if category_col else []
        for word in answer.split(' '):
            arr.append(to_num_dict[word])
        encoded_answers.append(arr)

    max_length = max([len(answer) for answer in encoded_answers])
    padded_answers = pad_sequences(encoded_answers, maxlen=max_length, padding='post').tolist()
    padded_answers = np.array(padded_answers).astype('float')
    return max_length, padded_answers, from_num_dict, to_num_dict

def decode_answers(encoded_answers, from_num_dict):

    decoded_answers = []
    for index, answer in enumerate(encoded_answers):
        arr = [index, answer[0]]
        for i, word in enumerate(answer):
            if i > 0 and word != 0:
                arr.append(from_num_dict[word])
        decoded_answers.append(arr)
    return decoded_answers

def generate_data(answer_df, subset=1, category_col=None):

    if category_col:
         max_length, encoded_answers, from_num_dict, to_num_dict = encode_answers(answer_df[answer_col].values,
                                                                             category_col, answer_df['id'])
    else:
        max_length, encoded_answers, from_num_dict, to_num_dict = encode_answers(answer_df[answer_col].values)

    encoded_answer_df = pd.DataFrame(encoded_answers)
    encoded_answer_df[predictor_col] = answer_df[predictor_col].astype(float)

    # randomize data
    randomized_data = encoded_answer_df.reindex(np.random.permutation(encoded_answer_df.index))
    max = int(len(randomized_data) * subset)
    randomized_labels = randomized_data[predictor_col].values[:max]
    randomized_answers = randomized_data.drop([predictor_col], axis=1).values[:max]

    return randomized_answers, randomized_labels, max_length, from_num_dict, to_num_dict

