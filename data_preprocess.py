from xml.dom import minidom
import pandas as pd

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
    # print(df)
    return question, reference_answer, answers_df

