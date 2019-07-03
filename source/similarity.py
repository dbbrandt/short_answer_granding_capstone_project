from sklearn.feature_extraction.text import CountVectorizer


# Calculate the ngram containment for one answer file/source file pair in a df
def calculate_containment(questions, answers, n_array):
    '''Calculates the containment between a given answer text and its associated source text.
       This function creates a count of ngrams (of a size, n) for each text file in our data.
       Then calculates the containment by finding the ngram count for a given answer text,
       and its associated source text, and calculating the normalized intersection of those counts.
       :param questions: A dataframe representing the test questions with columns,
           'id', 'answer'
       :param answers: A dataframe representing the graded student responses to questions with columns,
           'id', 'answer',
       :param n_array: An array of integers that defines the ngram sizes
       :return: Containment values for each answer and value of n that represents the similarity
           between an correcdt answer and stuent answer.
    '''

    ngrams_calc = []
    for index, answer_row in answers.iterrows():
        ngrams = []
        for n in n_array:
            containment = 0
            answer = answer_row.answer
            correct_answer = questions[questions.id == answer_row.id].iloc[0].answer
            print(f"index: {index} - answer: {answer}")

            # Calculate ngrams for n
            if len(answer) and len(correct_answer):
                counts = CountVectorizer(analyzer='word', ngram_range=(n, n))
                try:
                    ngram_array = counts.fit_transform([answer, correct_answer]).toarray()
                    vocabulary = counts.vocabulary_
                    # Calculate containment by getting the intersection of the ngrams for the answer and original
                    ngram_a = ngram_array[0]
                    ngram_o = ngram_array[1]
                    intersect = 0
                    # Iterate through the vocabulary getting the minimum ngram count from the two ngram sets.
                    # The minimum will return 0 if one of the ngrams doesn't contain the vocaulary, otherwise it will give
                    # the smallest number of times both original and answer have the ngram.
                    for key, i in vocabulary.items():
                        intersect += min(ngram_a[i], ngram_o[i])

                    if ngram_a.sum() > 0:
                        containment = intersect / ngram_a.sum()
                except:
                    print(f'Index: {index} - Faild to generate count vector')

            ngrams.append(containment)

        # add correct answer for analysis
        ngrams.append(answer_row.correct)
        ngrams_calc.append(ngrams)
    return ngrams_calc

def calculate_average_containment(n, df):
    category_vals = np.zeros(4)
    containment_vals = np.zeros(4)
    for i in df.index:
        # get level of plagiarism for a given file index
        category = df.loc[i, 'Category']
        category_vals[category] += 1
        # calculate containment for given file and n
        filename = df.loc[i, 'File']
        c = calculate_containment(df, n, filename)
        containment_vals[category] += c
    return containment_vals / category_vals

def show_average_containment(complete_df):
    results = {'all': 0, 'train': 0, 'test': 0}
    results['all'] = calculate_average_containment(n, complete_df)
    train_df = complete_df.loc[complete_df['Datatype'] != 'test']
    results['train'] = calculate_average_containment(n, train_df)
    test_df = complete_df.loc[complete_df['Datatype'] != 'train']
    results['test'] = calculate_average_containment(n, test_df)

    print(f"ngrams: {n}")
    for key, value in results.items():
        print(f"datatype: {key} result: {value}")
    print()


def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''

    # Remove extra spaces before creating word lists
    answer_list = ' '.join(answer_text.split()).split(' ')
    source_list = ' '.join(source_text.split()).split(' ')

    a_len = len(answer_list)
    s_len = len(source_list)

    # Array to store the dynamic programming results of LCS calculation
    lcs_array = np.zeros((s_len + 1, a_len + 1))

    for row, row_word in enumerate(source_list):
        for col, col_word in enumerate(answer_list):
            # If this original word matches the answer word, calculate the lcs_array value (offset by 1 row, 1 col)
            if row_word == col_word:
                lcs_array[row + 1, col + 1] = lcs_array[row, col] + 1
            else:
                # If not a match sets the current (offset) cell to the max of row or column value
                lcs_array[row + 1, col + 1] = max(lcs_array[row + 1, col], lcs_array[row, col + 1])

    return lcs_array[s_len, a_len] / a_len


# Function returns a list of containment features, calculated for a given n
# Should return a list of length 100 for all files in a complete_df
def create_containment_features(df, n, column_name=None):
    containment_values = []

    if (column_name == None):
        column_name = 'c_' + str(n)  # c_1, c_2, .. c_n

    # iterates through dataframe rows
    for i in df.index:
        file = df.loc[i, 'File']
        # Computes features using calculate_containment function
        if df.loc[i, 'Category'] > -1:
            c = calculate_containment(df, n, file)
            containment_values.append(c)
        # Sets value to -1 for original tasks
        else:
            containment_values.append(-1)

    print(str(n) + '-gram containment features created!')
    return containment_values


# Function creates lcs feature and add it to the dataframe
def create_lcs_features(df, column_name='lcs_word'):
    lcs_values = []

    # iterate through files in dataframe
    for i in df.index:
        # Computes LCS_norm words feature using function above for answer tasks
        if df.loc[i, 'Category'] > -1:
            # get texts to compare
            answer_text = df.loc[i, 'Text']
            task = df.loc[i, 'Task']
            # we know that source texts have Class = -1
            orig_rows = df[(df['Class'] == -1)]
            orig_row = orig_rows[(orig_rows['Task'] == task)]
            source_text = orig_row['Text'].values[0]

            # calculate lcs
            lcs = lcs_norm_word(answer_text, source_text)
            lcs_values.append(lcs)
        # Sets to -1 for original tasks
        else:
            lcs_values.append(-1)

    print('LCS features created!')
    return lcs_value

def generate_ngrams():
    # Define an ngram range
    ngram_range = range(1, 10)

    # The following code may take a minute to run, depending on your ngram_range
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    features_list = []

    # Create features in a features_df
    all_features = np.zeros((len(ngram_range) + 1, len(complete_df)))

    # Calculate features for containment for ngrams in range
    i = 0
    for n in ngram_range:
        column_name = 'c_' + str(n)
        features_list.append(column_name)
        # create containment features
        all_features[i] = np.squeeze(create_containment_features(complete_df, n))
        i += 1

    # Calculate features for LCS_Norm Words
    features_list.append('lcs_word')
    all_features[i] = np.squeeze(create_lcs_features(complete_df))

    # create a features dataframe
    features_df = pd.DataFrame(np.transpose(all_features), columns=features_list)

    # Print all features/columns
    print()
    print('Features: ', features_list)

    corr_matrix = features_df.corr().abs().round(2)
    # display shows all of a dataframe
    display(corr_matrix)
