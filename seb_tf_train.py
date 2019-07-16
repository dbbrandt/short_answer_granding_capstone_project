from source.utils import train_and_test, load_seb_data, save_results

def main():
    model_dir = 'model/seb'
    model_file = 'baseline'
    questions_file = f"data/seb/questions.csv"
    results_dir = 'data/results/seb'
    train = True
    pretrained = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_seb_data(pretrained)

    # For preptained, the vocabulary is some subset of the full Glove dataset. We need the highest key
    # used because the embedding layer assumes a continuous set of values.
    vocab_size = len(vocabulary)

    model_params = {'max_answer_len': max_answer_len,
                    'vocab_size': vocab_size,
                    'epochs': 1000,
                    'pretrained': pretrained,
                    'embedding_dim': 50,
                    'flatten': True,
                    'lstm_dim_1': 100,
                    'lstm_dim_2': 0,
                    'dropout': 0.2}

    eval, results = train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test,
                   vocabulary, questions_file, train, False)

    save_results(f"{results_dir}/seb_tf_train", model_params, eval, results)

main()

