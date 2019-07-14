import datetime
from xml.dom import minidom
import pandas as pd
import numpy as np
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

from source.glove import Glove


answer_col = 'answer'
predictor_col = 'correct'
sag_question_id_file = 'data/sag2/files'

glove = Glove()

def get_glove():
    global glove
    if not glove.loaded:
        glove = Glove(True)
    return glove

def get_xml_elements(doc, tag):
    ''' Helper method to extract the desired string value based on a tag from the SEB XML
        Used by read_xml_qanda().
    '''
    elements = doc.getElementsByTagName(tag)
    values = [e.firstChild for e in elements]
    return values, elements

def read_xml_qanda(filename, id):
    ''' SEB data specific method to extract key question and answer data
        Used by the load_seb_data() function.
    '''
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
    # text = re.sub(r"[^a-zA-Z0-9\-]", " ", text.lower())  # Convert to lower case
    # Removed "-" to better leverage the pretrained vocabulary
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Convert to lower case

    words = text.split()  # Split string into words
    # words = [w for w in words if w not in stopwords.words("english")]  # Remove stopwords
    # words = [PorterStemmer().stem(w) for w in words]  # stem
    return ' '.join(words)

def encode_answers(answer_df, pretrained=False, category_col=None, question_ids=[]):
    answers = [simplified_answer(answer) for answer in answer_df[answer_col].values]
    # answers = [answer for answer in answer_df[answer_col].values]
    vocab_string = " ".join(answers)
    word_list = vocab_string.split()
    word_list = sorted(list(dict.fromkeys(word_list)))
    #vocabulary = {i: word for i, word in enumerate(vocabulary)}

    if pretrained:
        encoded_answers, vocabulary = pretrained_encoded(answers, word_list, category_col, question_ids)
        print(f"Pretrained encoded_answers shape: {np.array(encoded_answers).shape}")
    else:
        to_encoded = {word: i for i, word in enumerate(word_list)}
        vocabulary = {i: word for i, word in enumerate(word_list)}
        encoded_answers = encoded(answers, to_encoded, category_col, question_ids)
        print(f"Simple encoded_answers shape: {np.array(encoded_answers).shape}")

    max_length = max([len(answer) for answer in encoded_answers])
    padded_answers = pad_sequences(encoded_answers, maxlen=max_length, dtype='float', padding='post').tolist()
    padded_answers = np.array(padded_answers)
    return max_length, padded_answers, vocabulary

def encoded(answers, to_encoded, category_col, question_ids):
    encoded_answers = []
    for index, answer in enumerate(answers):
        arr = [question_ids[index]] if category_col else []

        for word in answer.split(' '):
            arr.append(to_encoded[word])
        encoded_answers.append(arr)
    return encoded_answers

def pretrained_encoded(answers, word_list, category_col, question_ids):
    # instantiate and laod a Glove object
    glove = get_glove()
    word2index = glove.word2index
    found_words = [word for word in word_list if word in word2index]

    encoded_answers = []
    vocabulary = {}
    for index, answer in enumerate(answers):
        arr = [int(question_ids[index])] if category_col else []
        for word in answer.split(' '):
            if word in word2index:
                # Create a mapping to the glove embedding word index
                i = word2index[word]
                vocabulary[i] = word
                # Putpulate a custom embedding matrix with the words used indexed to the actual used vocabulary size.
                i = found_words.index(word)
                arr.append(i)
            else:
                print(f"ERROR! Word: {word} not found in dictionary")
                arr.append(0)

        encoded_answers.append(arr)

    # If using pretrained embeddings, load them based on the vocabulary
    glove.load_custom_embedding(vocabulary)

    return encoded_answers, vocabulary

def decode_answers(encoded_answers, from_encoded, pretrained=False):

    decoded_answers = []
    for index, answer in enumerate(encoded_answers):
        arr = [answer[0]]
        for i, word in enumerate(answer):
            if i > 0 and word != 0:
                if pretrained:
                    hash = Glove(word)
                    arr.append(from_encoded[hash])
                else:
                    arr.append(from_encoded[word])

        decoded_answers.append(arr)
    return decoded_answers

def decode_predictions(X_test, y_test, vocabulary, prediction, questions_file):
    ''' Combine the prediction results with the original answer data
        Return: Dataframe of human readable predictions and related string data
    '''
    questions = pd.read_csv(questions_file, dtype={'id':str})
    decoded = decode_answers(X_test, vocabulary)
    results = []
    for index, answer in enumerate(decoded):
        answer_text = ' '.join(answer[1:])
        correct = y_test[index]
        question = questions.iloc[int(answer[0])]
        pred = prediction[index]
        results.append([index, question.id, answer_text, question.answer, correct, pred])
    return pd.DataFrame(results, columns=['test_id', 'question_id', 'answer', 'correct_answer', 'correct', 'prediction'])


def generate_data(answer_df, pretrained=False, sample_size=1, question_id=None):
    ''' Convert the provided asnwers into numerically encoded data'''

    if question_id:
         max_length, encoded_answers, vocabulary = encode_answers(answer_df, pretrained, question_id,
                                                                                  answer_df[question_id])
    else:
         max_length, encoded_answers, vocabulary = encode_answers(answer_df, pretrained, question_id,
                                                                                  answer_df[question_id])

    encoded_answer_df = pd.DataFrame(encoded_answers)
    encoded_answer_df[predictor_col] = answer_df[predictor_col].astype(float)

    # randomize data
    randomized_data = encoded_answer_df.reindex(np.random.permutation(encoded_answer_df.index))
    # max will allow sample size less the 100% of data.
    max = int(len(randomized_data) * sample_size)
    randomized_labels = randomized_data[predictor_col].values[:max]
    randomized_answers = randomized_data.drop([predictor_col], axis=1).values[:max]

    return randomized_answers, randomized_labels, max_length, vocabulary

def load_seb_data(pretrained=False, sample_size=1, verbose=False):
    filenames = {'em': 'data/source_data/sciEntsBank/EM-post-35.xml',
                 'me': 'data/source_data/sciEntsBank/ME-inv1-28b.xml',
                 'mx': 'data/source_data/sciEntsBank/MX-inv1-22a.xml',
                 'ps': 'data/source_data/sciEntsBank/PS-inv3-51a.xml'}


    seed(72) # Python
    set_random_seed(72) # Tensorflow

    # question, reference_answer, answer_df = read_xml_qanda(filenames['ps'])
    questions = []
    answer_df = pd.DataFrame()
    for index, filename in enumerate(filenames.values()):
        answers, question = read_xml_qanda(filename, str(index))
        answer_df = pd.concat([answer_df, answers], axis=0, ignore_index=True)
        questions.append(question)

    if verbose:
        print(answer_df)
    answer_df.to_csv('data/seb/answers.csv', index=False)

    questions_df = pd.DataFrame(questions, columns=['question','answer'])
    questions_df['id'] = questions_df.index
    questions_df.to_csv('data/seb/questions.csv', index=False)

    data_answers, data_labels, max_length, vocabulary = generate_data(answer_df, pretrained, sample_size, 'id')

    if verbose:
        print(data_answers)
        print(data_labels)

    print(f"Sample Size: {len(data_answers)}")
    print(f"Longest answer: {max_length}")
    print(f"Vocabulary Size:{len(vocabulary)}")
    # Save Vocabulary
    pd.DataFrame(vocabulary.values()).to_csv('data/seb/vocab.csv', index=False, header=None)

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

    if verbose:
        raw_answers = decode_answers(test_data.values, vocabulary)
        for answer in raw_answers:
            print(f"{answer[0]}. correct: {answer[1]} answer: {' '.join(answer[2:])}")

    return X_train, y_train, X_test, y_test, max_length, vocabulary


def str_id_map(filename):
    ''' Function to create a map of arbitrary question identifiers (string) to numbers
        The SAG data has a X.Y notation for quetions which are non-numerical ex. 12.10.
    '''
    ids = pd.read_csv(filename, dtype={'id': str})['id'].values
    id_to_num = {id: i for i, id in enumerate(ids)}
    return id_to_num

def load_sag_data(pretrained=False, percent_of_data=1, verbose=False):
    # filename = 'data/sag2/answers.csv'
    filename = 'data/sag2/balanced_answers.csv'

    seed(72) # Python
    set_random_seed(72) # Tensorflow

    answer_df = pd.read_csv(filename, dtype={'id': str})

    # We need to convert the question id which is text to a number for future identification of related question.
    id_to_num = str_id_map(sag_question_id_file)
    answer_df['id'] = answer_df['id'].apply(lambda a: id_to_num[a])

    if verbose:
        print(answer_df)

    data_answers, data_labels, max_length, vocabulary= generate_data(answer_df, pretrained, percent_of_data, 'id')
    if verbose:
        print(data_answers)
        print(data_labels)
    print(f"Sample Size: {len(data_answers)}")
    print(f"Longest answer: {max_length}")

    # Save Vocabulary
    pd.DataFrame(vocabulary.values()).to_csv('data/sag2/vocab.csv', index=False, header=None)

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

    if verbose:
        raw_answers = decode_answers(test_data.values, vocabulary)
        for answer in raw_answers:
            print(f"{answer[0]}. correct: {answer[1]} answer: {' '.join(answer[2:])}")

    return X_train, y_train, X_test, y_test, max_length, vocabulary

def build_model(params):
    model = Sequential()
    if params['pretrained']:
        model.add(Embedding(params['vocab_size'], params['embedding_dim'], weights=[glove.custom_embedding_matrix],
                            input_length=params['max_answer_len'], trainable=False))
    else:
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
    # fit the model
    model.fit(X_train, y_train, epochs=epochs, verbose=1) #, validation_data=(X_test, y_test))

    # evaluate the model
    print(f"test_answers shape: {X_train.shape}")
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % loss)

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
                   vocabulary, questions_file, train=False, verbose=False):

    if train:
        # Build Model
        model = build_model(model_params)
        fit_model(model, model_dir, model_params['epochs'], X_train, y_train)
    else:
        # Load existing model
        filename = f"{model_dir}/{model_file}"
        model = load_model(filename)
        print(model.summary())

    eval, results = evaluate(model, X_test, y_test)

    if verbose:
        results_df = decode_predictions(X_test, y_test, vocabulary, results['pred'].values, questions_file)
        print_results(results_df)

    return eval, results

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
    print(f"Min pred: {min_pred} max_pred: {max_pred}")
    if max_pred - min_pred > 0:
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

    features_df = pd.DataFrame(test_features)
    results_df = pd.DataFrame(features_df[0])
    results_df['test_y'] = test_labels
    results_df['predictions'] = test_preds

    results_df.columns = ['id','test_y', 'pred']

    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy }, results_df

def print_results(results_df, show_correct=False):

    incorrect = results_df[results_df['correct'] != results_df['prediction']]
    print("Incorrect Predictions")
    print("id, question_id, prediction, correct, answer, correct_answer")
    for index, row in incorrect.iterrows():
        print(f'{row.test_id},{row.question_id},{row.prediction},{row.correct},"{row.answer}","{row.correct_answer}"')

    if show_correct:
        correct = results_df[results_df['correct'] == results_df['prediction']]
        print("Correct Predictions")
        print("id, question_id, prediction, correct, answer, correct_answer")
        for index, row in incorrect.iterrows():
            print(f'{row.test_id},{row.question_id},{row.prediction},{row.correct},"{row.answer}","{row.correct_answer}"')

def save_results(filename, model_params, eval, results):
    now = datetime.datetime.now()
    results_file = open(f"{filename}_{now}.txt","w+")
    results_file.write(str(model_params))
    results_file.write(str(eval))
    results_file.close()

    results.to_csv(f"{filename}_pred_{now}.csv", index=False)