from source.utils import train_and_test, load_sag_data, save_results

def main():
    model_dir = 'model/sag'
    model_file = '100pct'  # used to load existing model if train is false
    questions_file = 'data/sag2/questions.csv'
    results_dir = 'data/results/sag'
    train = True
    # Pretrained embeddings
    pretrained = False
    data_percentage = 1

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_sag_data(pretrained, data_percentage)

    vocab_size = len(vocabulary)
    print(f"Vocabulary Size: {vocab_size}")

    model_params = {'max_answer_len': max_answer_len,
                    'vocab_size': vocab_size,
                    'epochs': 20,
                    'pretrained': pretrained,
                    'embedding_dim': 50,
                    'flatten': True,
                    'lstm_dim_1': 100,
                    'lstm_dim_2': 20,
                    'dropout': 0.3}

    # Trains model if train=True and prints out metrics on results (see below)
    eval, results = train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test,
                   vocabulary, questions_file, train)

    save_results(f"{results_dir}/sag_tf_train", model_params, eval, results)

main()


