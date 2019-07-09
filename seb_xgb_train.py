from source.utils import load_xgb_model, save_xgb_model, load_seb_data, evaluate, decode_predictions, print_results
from xgboost import XGBClassifier

def build_model(model_params):
    model = XGBClassifier(scale_pos_weight=1,
                          learning_rate=0.01,
                          objective=model_params['objective'],
                          subsample=0.8,
                          min_child_weight=6,
                          n_estimators=10000,
                          max_depth=10,
                          gamma=6,
                          verbosity=0)

    return model

def main():
    model_file = 'model/seb/xgboost/baseline'
    questions_file = 'data/seb/questions.csv'
    train = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_seb_data()

    # model_params = {'objective': 'binary:logistic'}
    # model_params = {'objective': 'reg:squarederror'}
    model_params = {'objective': 'binary:hinge'}
    if train:
        # Build Model
        model = build_model(model_params)
        model.fit(X_train, y_train)
        save_xgb_model(model, model_file)
    else:
        # Load existing model
        model = load_xgb_model(model_file)

    print(model)
    eval = evaluate(model, X_test, y_test)
    results_df = decode_predictions(X_test, y_test, vocabulary, eval['Predictions'], questions_file)
    print_results(results_df)

main()
