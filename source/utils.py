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
import joblib

answer_col = 'answer'
predictor_col = 'correct'
sag_question_id_file = 'data/sag2/files'

def get_xml_elements(doc, tag):
    elements = doc.getElementsByTagName(tag)
    values = [e.firstChild for e in elements]
    return values, elements

def read_xml_qanda(filename, id):
    doc = minidom.parse(filename)
    values, elements = get_xml_elements(doc, 'questionText')
    question_text = values[0].data
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
    answers_df['question'] = question_text
    answers_df['reference_answer'] = reference_answer
    answers_df['id'] = id
    return answers_df, [question_text, reference_answer]

def simplified_answer(answer):
    # nltk.download("stopwords", quiet=True)

    text = BeautifulSoup(answer, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\-]", " ", text.lower())  # Convert to lower case
    words = text.split()  # Split string into words
    # words = [w for w in words if w not in stopwords.words("english")]  # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words]  # stem

    return ' '.join(words)

def str_id_map(filename):
    ids = pd.read_csv(filename, dtype={'id': str})['id'].values
    id_to_num = {id: i for i, id in enumerate(ids)}
    return id_to_num

def encode_answers(answer_df, category_col=None, question_id=[], ngrams=False):
    answers = [simplified_answer(answer) for answer in answer_df[answer_col].values]
    vocab_string = " ".join(answers)
    vocab_list = vocab_string.split()
    vocabulary = sorted(list(dict.fromkeys(vocab_list)))
    to_num_dict = {word: i for i, word in enumerate(vocabulary)}
    from_num_dict = {i: word for i, word in enumerate(vocabulary)}

    encoded_answers = []
    for index, answer in enumerate(answers):
        arr = [question_id[index]] if category_col else []

        if ngrams:
            ng = list(answer_df.iloc[index][['ng1', 'ng2']].values)
            arr += ng

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
        arr = [answer[0]]
        for i, word in enumerate(answer):
            if i > 0 and word != 0:
                arr.append(from_num_dict[word])

        decoded_answers.append(arr)
    return decoded_answers

def decode_predictions(X_test, y_test, vocabulary, prediction, questions_file):
    questions = pd.read_csv(questions_file, index_col=0)
    decoded = decode_answers(X_test, vocabulary)
    results = []
    for index, answer in enumerate(decoded):
        answer_text = ' '.join(answer[1:])
        correct = y_test[index]
        correct_answer = questions.iloc[int(answer[0])].answer
        pred = prediction[index]
        results.append([index, answer_text, correct_answer, correct, pred])
    return pd.DataFrame(results, columns=['id', 'answer', 'correct_answer', 'correct', 'prediction'])


# Note: Adding in the question_id may help the learning to group relationsips better by the question..
# This needs to be validated. It can't hurt.
# Sample_size allow a franction of the data to be used for testing.
# Question_id is the columname of the field to use to identify the question
# Ngrams determines if we should append ngrams to the encoded data.
def generate_data(answer_df, sample_size=1, question_id=None, ngrams=False):

    if question_id:
         max_length, encoded_answers, from_num_dict, to_num_dict = encode_answers(answer_df, question_id,
                                                                                  answer_df[question_id], ngrams)
    else:
         max_length, encoded_answers, from_num_dict, to_num_dict = encode_answers(answer_df, question_id,
                                                                                  answer_df[question_id])

    encoded_answer_df = pd.DataFrame(encoded_answers)
    encoded_answer_df[predictor_col] = answer_df[predictor_col].astype(float)

    # randomize data
    randomized_data = encoded_answer_df.reindex(np.random.permutation(encoded_answer_df.index))
    max = int(len(randomized_data) * sample_size)
    randomized_labels = randomized_data[predictor_col].values[:max]
    randomized_answers = randomized_data.drop([predictor_col], axis=1).values[:max]

    return randomized_answers, randomized_labels, max_length, from_num_dict, to_num_dict


def load_seb_data(sample_size=1):
    filenames = {'em': 'data/sciEntsBank/EM-post-35.xml',
                 'me': 'data/sciEntsBank/ME-inv1-28b.xml',
                 'mx': 'data/sciEntsBank/MX-inv1-22a.xml',
                 'ps': 'data/sciEntsBank/PS-inv3-51a.xml'}


    seed(72) # Python
    set_random_seed(72) # Tensorflow

    # question, reference_answer, answer_df = read_xml_qanda(filenames['ps'])
    questions = []
    answer_df = pd.DataFrame()
    for index, filename in enumerate(filenames.values()):
        answers, question = read_xml_qanda(filename, str(index))
        answer_df = pd.concat([answer_df, answers], axis=0, ignore_index=True)
        questions.append(question)

    print(answer_df)
    answer_df.to_csv('data/seb/answers.csv', index=False)

    questions_df = pd.DataFrame(questions, columns=['question','answer'])
    questions_df.to_csv('data/seb/questions.csv')

    data_answers, data_labels, max_length, from_num_vocab, to_num_vocab = generate_data(answer_df, sample_size, 'id')

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

def load_sag_data(percent_of_data=1, ngrams=False):
    if ngrams:
        filename = 'data/sag2/answers_ngrams.csv'
    else:
        filename = 'data/sag2/answers.csv'
        # filename = 'data/sag2/balanced_answers.csv'

    seed(72) # Python
    set_random_seed(72) # Tensorflow

    answer_df = pd.read_csv(filename, dtype={'id': str})

    id_to_num = str_id_map(sag_question_id_file)
    answer_df['id'] = answer_df['id'].apply(lambda a: id_to_num[a])

    print(answer_df)

    data_answers, data_labels, max_length, from_num_dict, to_num_dict = generate_data(answer_df, percent_of_data,
                                                                                      'id', ngrams)

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

# Saving Keras Tensorflow models uses provided load and save
def load_model(model_dir):
    return load_keras_model(model_dir)

def save_model(model, model_dir):
    save_keras_model(model, model_dir)

# Saving XGB models uses joblib
def load_xgb_model(filename):
    return joblib.load(filename + '.xgb')

def save_xgb_model(model, filename):
    joblib.dump(model, filename)

def train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test,
                   vocabulary, questions_file, train=False):

    if train:
        # Build Model
        model = build_model(model_params)
        fit_model(model, model_dir, model_params['epochs'], X_train, y_train)
    else:
        # Load existing model
        filename = f"{model_dir}/{model_file}"
        model = load_model(filename)
        print(model.summary())

    eval = evaluate(model, X_test, y_test)

    results_df = decode_predictions(X_test, y_test, vocabulary, eval['Predictions'], questions_file)
    print_results(results_df)

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
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy, 'Predictions': test_preds}

def print_results(results_df):

    incorrect = results_df[results_df['correct'] != results_df['prediction']]
    print("Incorrect Predictions")
    print("id, prediction, correct, answer, correct_answer")
    for index, row in incorrect.iterrows():
        print(f'{row.id}, {row.prediction}, "{row.correct}", "{row.answer}", "{row.correct_answer}"')

    correct = results_df[results_df['correct'] == results_df['prediction']]
    print("Correct Predictions")
    print("id, prediction, correct, answer, correct_answer")
    for index, row in correct.iterrows():
        print(f'{row.id}, {row.prediction}, "{row.correct}", "{row.answer}", "{row.correct_answer}"')