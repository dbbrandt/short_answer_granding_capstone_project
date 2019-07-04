from source.utils import load_xgb_model, save_xgb_model, load_seb_data, evaluate, decode_predictions
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
    model_file = 'model/seb/xgboost/baseline'
    questions_file = 'data/seb/questions.csv'
    train = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_seb_data()

    model_params = {'objective': 'binary:logistic'}
    #model_params = {'objective': 'reg:squarederror'}
    #model_params = {'objective': 'binary:hinge'}
    if train:
        # Build Model
        model = build_model(model_params)
        model.fit(X_train, y_train)
        save_xgb_model(model, model_file)
    else:
        # Load existing model
        model = load_xgb_model(model_file)
        print(model)

    print(model)
    eval = evaluate(model, X_test, y_test)
#    results_df = decode_predictions(X_test, y_test, vocabulary, eval['Predictions'], questions_file)

#    incorrect = results_df[results_df['correct'] != results_df['prediction']]
#    print(incorrect)

main()

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=1000, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=1)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           19    6
# 1.0           12    5
#
# Recall:     0.294
# Precision:  0.455
# Accuracy:   0.571

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=1000, n_jobs=1,
#               nthread=None, objective='reg:squarederror', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=0)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           22    3
# 1.0           12    5
#
# Recall:     0.294
# Precision:  0.625
# Accuracy:   0.643

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=1000, n_jobs=1,
#               nthread=None, objective='binary:hinge', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=0)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           21    4
# 1.0            6   11
#
# Recall:     0.647
# Precision:  0.733
# Accuracy:   0.762