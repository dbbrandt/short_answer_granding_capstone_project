from source.utils import train_and_test, load_sag_data

def main():
    model_dir = 'model/sag'
    model_file = '20pct'  # used to load existing model if train is false
    train = True
    data_percentage = 0.2

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_sag_data(data_percentage)

    model_params = {'max_answer_len': max_answer_len,
                    'vocab_size': len(vocabulary),
                    'epochs': 30,
                    'embedding_dim': 50,
                    'flatten': True,
                    'lstm_dim_1': 100,
                    'lstm_dim_2': 20,
                    'dropout': 0.2}

    # Trains model if train=True and prints out metrics on results (see below)
    train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test, train)

main()

# =================================================================
# Results
# =================================================================
# Embedding: 200, Dropout: 0.2, LSTM: 200, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 72.7%
# Racal: 1   never predics false
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# embedding_1 (Embedding)      (None, 47, 30)            4050
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 47, 30)            0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               52400
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 56,551
# Trainable params: 56,551
# Non-trainable params: 0

# predictions  0.0  1.0
# actuals
# 0.0           21    4
# 1.0            9    8
#
# Recall:     0.471
# Precision:  0.667
# Accuracy:   0.690

# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 74%  Recal: 1
# Never predics false
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# _________________________________________________________________
# =================================================================
# embedding (Embedding)        (None, 175, 200)          392600
# _________________________________________________________________
# dropout (Dropout)            (None, 175, 200)          0
# _________________________________________________________________
# lstm (LSTM)                  (None, 200)               320800
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 200)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 201
# =================================================================
# Total params: 713,601
# Trainable params: 713,601
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 1709 samples, validate on 733 samples
# WARNING:tensorflow:From /Users/dbrandt/Dropbox/PycharmProjects/short_answer_granding_capstone_project/venv/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# 2019-06-27 14:03:44.621063: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# Epoch 1/10
# 1709/1709 [==============================] - 18s 11ms/sample - loss: 0.2302 - acc: 0.6969 - val_loss: 0.1998 - val_acc: 0.7244
# Epoch 2/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2032 - acc: 0.7262 - val_loss: 0.2005 - val_acc: 0.7244
# Epoch 3/10
# 1709/1709 [==============================] - 18s 10ms/sample - loss: 0.2027 - acc: 0.7262 - val_loss: 0.2020 - val_acc: 0.7244
# Epoch 4/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2016 - acc: 0.7262 - val_loss: 0.2097 - val_acc: 0.7244
# Epoch 5/10
# 1709/1709 [==============================] - 18s 11ms/sample - loss: 0.2063 - acc: 0.7262 - val_loss: 0.2019 - val_acc: 0.7244
# Epoch 6/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2042 - acc: 0.7256 - val_loss: 0.2029 - val_acc: 0.7244
# Epoch 7/10
# 1709/1709 [==============================] - 18s 10ms/sample - loss: 0.2055 - acc: 0.7262 - val_loss: 0.2023 - val_acc: 0.7244
# Epoch 8/10
# 1709/1709 [==============================] - 16s 10ms/sample - loss: 0.2029 - acc: 0.7267 - val_loss: 0.2034 - val_acc: 0.7244
# Epoch 9/10
# 1709/1709 [==============================] - 18s 10ms/sample - loss: 0.2051 - acc: 0.7244 - val_loss: 0.2073 - val_acc: 0.7244
# Epoch 10/10
# 1709/1709 [==============================] - 17s 10ms/sample - loss: 0.2035 - acc: 0.7262 - val_loss: 0.2006 - val_acc: 0.7244
# test_answers shape: (1709, 175)
# Accuracy: 72.615564
# Loss: 0.199718
# predictions  1.0
# actuals
# 0.0          202
# 1.0          531
#
# Recall:     1.000
# Precision:  0.724
# Accuracy:   0.724

# Embedding: 400, Dropout: 0.2, LSTM: 400, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Balanced data 1342 points 50/50 split correct/incorrect.
# Accuracy: 50.479233
# Loss: 0.261770
# predictions  0.0  1.0
# actuals
# 0.0            1  205
# 1.0            0  197
#
# Recall:     1.000
# Precision:  0.490
# Accuracy:   0.491

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Balanced data 40 points 50/50 split correct/incorrect.
# Loss: 0.483871
# predictions  1.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     1.000
# Precision:  0.585
# Accuracy:   0.585

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Added Learning Rate and Decay to Adam optimizer. learning_rate = 1e-3
# Accuracy: 51.612902
# Loss: 0.250435
# predictions  1.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     1.000
# Precision:  0.585
# Accuracy:   0.585



# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Switch from linear activation to sigmoid, switch from mean_squared_error to binary_crossentropy. This worked best for
# my other answer validation.
# Balanced data 40 points 50/50 split correct/incorrect.
# NOTE: All predictions are still the same at 51.3 which is possibly due to the fact that the padding makes most of the
# input data 0's due to the wide range of input lengths.
# Accuracy: 51.612902
# Loss: 0.692645
# predictions  1.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     1.000
# Precision:  0.585
# Accuracy:   0.585
#
# [[0.51313174]
#  [0.51313174]
#  [0.51313174]

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Switch back to Linear and mean_squared_error and added a Masking layer to mask 0's
# Balanced data 40 points 50/50 split correct/incorrect.
# Just shifted all the answers to false
# Accuracy: 48.387095
# Loss: 0.255791
# /Users/dbrandt/Dropbox/PycharmProjects/short_answer_granding_capstone_project/source/utils.py:61: RuntimeWarning: invalid value encountered in long_scalars
#   precision = tp / (tp + fp)
# predictions  0.0
# actuals
# 0.0           17
# 1.0           24
#
# Recall:     0.000
# Precision:  nan
# Accuracy:   0.415
#
# [[0.4383411 ]
#  [0.4383411 ]

# Embedding: 400, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 10, Train Tst Split: .3 => Accuracy: 49%
# Switch back to Linear and mean_squared_error and added a Masking layer to mask 0's
# Balanced data 40 points 50/50 split correct/incorrect.
# Just shifted all the answers to false

# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 20, Train Tst Split: .3 => Accuracy: 65.9%
# Added in Flattening and Shaping after Embedding Layer as suggested by documentation for Embedding.
# Small Dataset : Train on 93 samples, validate on 41 samples
# Normalized predictions
# Accuracy: 100.000000
# Loss: 0.011860
# Min preds: 0.18662022054195404 max_preds: 0.6710594892501831
# predictions  0.0  1.0
# actuals
# 0.0           12    5
# 1.0            9   15
#
# Recall:     0.625
# Precision:  0.750
# Accuracy:   0.659
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 63.4%
# Added in Flattening and Shaping after Embedding Layer as suggested by documentation for Embedding.
# Small Dataset : Train on 93 samples, validate on 41 samples
# Normalized predictions
# Accuracy: 100.000000
# Loss: 0.001300
# Min preds: 0.020597627386450768 max_preds: 0.7078230977058411
# predictions  0.0  1.0
# actuals
# 0.0            7   10
# 1.0            5   19
#
# Recall:     0.792
# Precision:  0.655
# Accuracy:   0.634
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 56.8%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Accuracy: 100.000000
# Loss: 0.003250
# Min preds: 0.0336155965924263 max_preds: 0.9188557863235474
# predictions  0.0  1.0
# actuals
# 0.0           25   15
# 1.0           20   21
#
# Recall:     0.512
# Precision:  0.583
# Accuracy:   0.568
# ==================================
# Embedding: 100, Dropout: 0.2, LSTM 1: 100, LSTM 2: 20, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 65.4%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Increase embedding to 100
# Accuracy: 100.000000
# Loss: 0.002413
# Min preds: 0.0002403445541858673 max_preds: 0.983460009098053
# predictions  0.0  1.0
# actuals
# 0.0           31    9
# 1.0           19   22
#
# Recall:     0.537
# Precision:  0.710
# Accuracy:   0.654
# ==================================
# Embedding: 200, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 20, Train Tst Split: .3 => Accuracy: 65.4%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Increase embedding to 200, reduce epocs to 20 as previous stabilized at about 15
# Note: Not converging but demonstrates that the previous run was not helping.
# Run with 50 embeddings had a better range of predicts.
# Accuracy: 49.197862
# Loss: 0.245283
# Min preds: 0.47573280334472656 max_preds: 0.48620232939720154
# predictions  0.0  1.0
# actuals
# 0.0           29   11
# 1.0           17   24
#
# Recall:     0.585
# Precision:  0.686
# Accuracy:   0.654
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 20, Train Tst Split: .3 => Accuracy: 56.8%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Decrease embedding back to 50, reduce epocs to 20 as previous stabilized at about 15
# Accuracy: 100.000000
# Loss: 0.004790
# Min preds: 0.15179654955863953 max_preds: 1.0073564052581787
# predictions  0.0  1.0
# actuals
# 0.0           30   10
# 1.0           25   16
#
# Recall:     0.390
# Precision:  0.615
# Accuracy:   0.568
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 200, optmizer: Adam, epochs: 20, Train Tst Split: .3 => Accuracy: 56.8%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Increase LSTM to 200
# Accuracy: 100.000000
# Loss: 0.003766
# Min preds: 0.14939424395561218 max_preds: 0.9931410551071167
# predictions  0.0  1.0
# actuals
# 0.0           31    9
# 1.0           26   15
#
# Recall:     0.366
# Precision:  0.625
# Accuracy:   0.568
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 200, optmizer: Adam, epochs: 20, Train Tst Split: .3 => Accuracy: 68%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Switch from balanced to actual data which is 1/3 incorrect 2/3 correct.
# Note; This imporved the balance of correct and incorrect.
# Accuracy: 100.000000
# Loss: 0.002546
# Min preds: -0.07342652231454849 max_preds: 1.2405585050582886
# predictions  0.0  1.0
# actuals
# 0.0           23   20
# 1.0           27   77
#
# Recall:     0.740
# Precision:  0.794
# Accuracy:   0.680
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 20, Train Tst Split: .3 => Accuracy: 69.4%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Using acutal (not balanced data).
# Reduced LSTM back to 100
# Accuracy: 100.000000
# Loss: 0.009522
# Min preds: -0.12270881980657578 max_preds: 1.2196975946426392
# predictions  0.0  1.0
# actuals
# 0.0           25   18
# 1.0           27   77
#
# Recall:     0.740
# Precision:  0.811
# Accuracy:   0.694
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 100, LSTM 2: 20, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 72.8%
# Move to 20% of data: Train on 187 samples, validate on 81 samples
# Using acutal (not balanced data).
# Added a second layer of LSTM, increased Epocs to 50
# Note: Squed results to Recal.

# Accuracy: 100.000000
# Loss: 0.002845
# Min preds: -0.022228777408599854 max_preds: 0.9874494075775146
# predictions  0.0  1.0
# actuals
# 0.0           13   30
# 1.0           10   94
#
# Recall:     0.904
# Precision:  0.758
# Accuracy:   0.728

# ==================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 100, LSTM 2: 20, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 58.0%
# 20% of data: Train on 187 samples, validate on 81 samples
# Using balanced data to increase flase predictions.
# Second layer of LSTM, increased Epocs to 50
# Note: Drastically reduced true positives.
# Accuracy: 100.000000
# Loss: 0.001701
# Min preds: 0.005335822701454163 max_preds: 1.017879605293274
# predictions  0.0  1.0
# actuals
# 0.0           23   17
# 1.0           17   24
#
# Recall:     0.585
# Precision:  0.585
# Accuracy:   0.580

# ==================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 100, LSTM 2: 50, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 70.1%
# 20% of data: Train on 187 samples, validate on 81 samples
# Second layer of LSTM, increased Epocs to 50
# Min preds: -0.007130637764930725 max_preds: 1.0701065063476562
# predictions  0.0  1.0
# actuals
# 0.0           13   30
# 1.0           14   90
#
# Recall:     0.865
# Precision:  0.750
# Accuracy:   0.701
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 50, LSTM 2: 50, optmizer: Adam, epochs: 50, Train Tst Split: .3 => Accuracy: 71.4%
# 20% of data: Train on 187 samples, validate on 81 samples
# Reduced First layer of LSTM
# Accuracy: 100.000000
# Loss: 0.000561
# Min preds: -0.012730248272418976 max_preds: 1.0793601274490356
# predictions  0.0  1.0
# actuals
# 0.0           14   29
# 1.0           13   91
#
# Recall:     0.875
# Precision:  0.758
# Accuracy:   0.714

# ==================================
# Refactored model build out of main
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 100, LSTM 2: 20, optmizer: Adam, epochs: 30, Train Tst Split: .3 => Accuracy: 71.4%
# 20% of data: Train on 187 samples, validate on 81 samples
## Lowered epocs to 30 which was stable
# Results were the same.


# ==================================
# Ran saved model on full dataset and save model from best run 71.4%
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 100, LSTM 2: 20, optmizer: Adam, epochs: 30, Train Tst Split: .3 => Accuracy: 76.8%
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 176, 50)           98150
# _________________________________________________________________
# dropout (Dropout)            (None, 176, 50)           0
# _________________________________________________________________
# flatten (Flatten)            (None, 8800)              0
# _________________________________________________________________
# reshape (Reshape)            (None, 1, 8800)           0
# _________________________________________________________________
# lstm (LSTM)                  (None, 1, 100)            3560400
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 20)                9680
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 20)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 21
# =================================================================
# Total params: 3,668,251
# Trainable params: 3,668,251
# Non-trainable params: 0
# _________________________________________________________________
# None
# Min preds: -0.06254398822784424 max_preds: 1.1257044076919556
# predictions  0.0  1.0
# actuals
# 0.0           83  119
# 1.0           51  480
#
# Recall:     0.904
# Precision:  0.801
# Accuracy:   0.768

# =================================================================
# Train using full data (70/30 split)
# =================================================================
# Embedding: 50, Dropout: 0.2, LSTM 1: 100, LSTM 2: 20, optmizer: Adam, epochs: 30, Train Tst Split: .3 => Accuracy: 76.7%
# 100% of data:
# Note: Same results as running off 20% data training. Slight higher precision

# predictions  0.0  1.0
# actuals
# 0.0          102  100
# 1.0           71  460
#
# Recall:     0.866
# Precision:  0.821
# Accuracy:   0.767

# =================================================================
# Restuls of predict on full data set (train and test)
# =================================================================
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 176, 50)           98150
# _________________________________________________________________
# dropout (Dropout)            (None, 176, 50)           0
# _________________________________________________________________
# flatten (Flatten)            (None, 8800)              0
# _________________________________________________________________
# reshape (Reshape)            (None, 1, 8800)           0
# _________________________________________________________________
# lstm (LSTM)                  (None, 1, 100)            3560400
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 20)                9680
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 20)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 21
# =================================================================
# Total params: 3,668,251
# Trainable params: 3,668,251
# Non-trainable params: 0
# _________________________________________________________________
# None
# Min preds: -0.06451256573200226 max_preds: 1.1399285793304443
# predictions  0.0   1.0
# actuals
# 0.0          275   396
# 1.0          155  1616
#
# Recall:     0.912
# Precision:  0.803
# Accuracy:   0.774