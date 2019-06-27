from xml.dom import minidom
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup

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
    test_preds = np.squeeze(np.round(predictor.predict(test_features)))

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

    max_length, encoded_answers, from_num_dict, to_num_dict = encode_answers(answer_df[answer_col].values,
                                                                             category_col, answer_df['id'])

    encoded_answer_df = pd.DataFrame(encoded_answers)
    encoded_answer_df[predictor_col] = answer_df[predictor_col].astype(float)

    # randomize data
    randomized_data = encoded_answer_df.reindex(np.random.permutation(encoded_answer_df.index))
    max = int(len(randomized_data) * subset)
    randomized_labels = randomized_data[predictor_col].values[:max]
    randomized_answers = randomized_data.drop([predictor_col], axis=1).values[:max]

    return randomized_answers, randomized_labels, max_length, from_num_dict, to_num_dict

