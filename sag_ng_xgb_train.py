from source.utils import load_model, save_model, load_sag_ng_data, evaluate
from xgboost import XGBClassifier

def build_model(model_params):
    model = XGBClassifier(scale_pos_weight=1,
                          learning_rate=0.01,
                          objective=model_params['objective'],
                          subsample=0.8,
                          min_child_weight=6,
                          n_estimators=10000,
                          max_depth=4,
                          gamma=1,
                          verbosity=0)

    return model

def main():
    model_dir = 'model/seb'
    model_file = 'xgboost'
    train = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_sag_ng_data()

    model_params = {'objective': 'binary:logistic'}
    # model_params = {'objective': 'reg:squarederror'}
    # model_params = {'objective': 'binary:hinge'}
    if train:
        # Build Model
        model = build_model(model_params)
        model.fit(X_train, y_train)
#        save_model(model, model_dir)
    else:
        # Load existing model
        filename = f"{model_dir}/{model_file}"
        model = load_model(filename)
        # print(model.summary())

    print(model)
    evaluate(model, X_test, y_test)

main()


