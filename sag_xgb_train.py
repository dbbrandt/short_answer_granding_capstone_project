from source.utils import load_model, save_model, load_sag_data, evaluate
from xgboost import XGBClassifier

def build_model(model_params):
    model = XGBClassifier(scale_pos_weight=1,
                          learning_rate=0.01,
                          objective=model_params['objective'],
                          subsample=0.8,
                          min_child_weight=6,
                          n_estimators=1000,
                          max_depth=4,
                          gamma=1,
                          verbosity=0)

    return model

def main():
    model_dir = 'model/seb'
    model_file = 'xgboost'
    train = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_sag_data()

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


# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=1000, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=0)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           29  173
# 1.0           10  521
#
# Recall:     0.981
# Precision:  0.751
# Accuracy:   0.750

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=2000, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=0)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           49  153
# 1.0           16  515
#
# Recall:     0.970
# Precision:  0.771
# Accuracy:   0.769

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=4000, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=0)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           68  134
# 1.0           26  505
#
# Recall:     0.951
# Precision:  0.790
# Accuracy:   0.782

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1,
#               learning_rate=0.01, max_delta_step=0, max_depth=4,
#               min_child_weight=6, missing=None, n_estimators=10000, n_jobs=1,
#               nthread=None, objective='binary:logistic', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#               silent=None, subsample=0.8, verbosity=0)
# Min preds: 0.0 max_preds: 1.0
# predictions  0.0  1.0
# actuals
# 0.0           71  131
# 1.0           34  497
#
# Recall:     0.936
# Precision:  0.791
# Accuracy:   0.775


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
# 0.0           13  189
# 1.0            8  523
#
# Recall:     0.985
# Precision:  0.735
# Accuracy:   0.731


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
# 0.0           24  178
# 1.0           12  519
#
# Recall:     0.977
# Precision:  0.745
# Accuracy:   0.741