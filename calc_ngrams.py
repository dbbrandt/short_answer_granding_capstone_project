import pandas as pd
from similarity import calculate_containment

answers = pd.read_csv('data/sag2/answers.csv', dtype={'id':str})
questions = pd.read_csv('data/sag2/questions.csv', dtype={'id':str})

n = [1,2]
ngrams  = calculate_containment(questions, answers, n)

df = pd.DataFrame(ngrams, columns=['1','2','correct'])
df.to_csv('data/sag2/answer_ngrams.csv', index=False)

