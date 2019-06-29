from source.utils import train_and_test, load_seb_data

def main():
    model_dir = 'model/seb'
    model_file = 'baseline'
    train = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_seb_data()

    model_params = {'max_answer_len': max_answer_len,
                    'vocab_size': len(vocabulary),
                    'epochs': 200,
                    'embedding_dim': 50,
                    'flatten': False,
                    'lstm_dim_1': 100,
                    'lstm_dim_2': 0,
                    'dropout': 0.2}

    train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test, train)

main()

# =================================================================
# Test Results
# =================================================================
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 72.7%
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
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

# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 76.2%
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 47, 30)            7020
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 47, 30)            0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100)               52400
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 59,521
# Trainable params: 59,521
# Non-trainable params: 0

# Accuracy: 97.959184
# Loss: 0.020474

# predictions  0.0  1.0
# actuals
# 0.0           22    3
# 1.0            7   10
#
# Recall:     0.588
# Precision:  0.769
# Accuracy:   0.762

# _________________________________________________________________
# Rerun - Note lower train results in better test accuracy. Possible
# overfitting issue with training.
# _________________________________________________________________
#
# test_answers shape: (98, 47)
# Accuracy: 93.877551
# Loss: 0.055575
# predictions  0.0  1.0
# actuals
# 0.0           20    5
# 1.0            5   12
#
# Recall:     0.706
# Precision:  0.706
# Accuracy:   0.762


# _________________________________________________________________
# Rerun - Reduce Epochs to see if reducing trianing helps...
#
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 100, Train Tst Split: .3 => Accuracy: 73.8%
# _________________________________________________________________
# Accuracy: 89.795918
# Loss: 0.088208
# predictions  0.0  1.0
# actuals
# 0.0           22    3
# 1.0            8    9
#
# Recall:     0.529
# Precision:  0.750
# Accuracy:   0.738

# _________________________________________________________________
# Rerun - Reduce Epochs to see if reducing trianing helps...
#
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 150, Train Tst Split: .3 => Accuracy: 73.8%
# _________________________________________________________________
# test_answers shape: (98, 47)
# Accuracy: 91.836735
# Loss: 0.077527
# predictions  0.0  1.0
# actuals
# 0.0           23    2
# 1.0            9    8
#
# Recall:     0.471
# Precision:  0.800
# Accuracy:   0.738


# _________________________________________________________________
# Rerun - Original
#
# Embedding: 30, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 66.7%
# _________________________________________________________________
# Accuracy: 85.714286
# Loss: 0.123627
# predictions  0.0  1.0
# actuals
# 0.0           24    1
# 1.0           13    4
#
# Recall:     0.235
# Precision:  0.800
# Accuracy:   0.667


# _________________________________________________________________
# Rerun - Increase LSTM nodes
#
# Embedding: 30, Dropout: 0.2, LSTM: 200, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 66.7%
# _________________________________________________________________

# Accuracy: 88.775510
# Loss: 0.096415
# predictions  0.0  1.0
# actuals
# 0.0           23    2
# 1.0           12    5
#
# Recall:     0.294
# Precision:  0.714
# Accuracy:   0.667

# _________________________________________________________________
# Rerun - Decrease LSTM nodes
#
# Embedding: 30, Dropout: 0.2, LSTM: 50, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 71.4%
# _________________________________________________________________
# Accuracy: 100.000000
# Loss: 0.000954

# predictions  -0.0   1.0
# actuals
# 0.0            22     3
# 1.0             9     8
#
# Recall:     0.471
# Precision:  0.727
# Accuracy:   0.714
#=======================
# 6/28/19
#=======================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 76.2%
# Note: Stabalized at 76.2% at 194 epochs
# Also min and max are not greater than range 0 to 1. This should be normalized. -0.28 - 1.01
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 47, 50)            11700
# _________________________________________________________________
# dropout (Dropout)            (None, 47, 50)            0
# _________________________________________________________________
# lstm (LSTM)                  (None, 100)               60400
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 101
# =================================================================
# Total params: 72,201
# Trainable params: 72,201
# Non-trainable params: 0
# _________________________________________________________________
# None
# Train on 98 samples, validate on 42 samples
# Accuracy: 100.000000
# Loss: 0.000223
# Min preds: -0.28002065420150757 max_preds: 1.016097068786621
# predictions  0.0  1.0
# actuals
# 0.0           22    3
# 1.0            7   10
#
# Recall:     0.588
# Precision:  0.769
# Accuracy:   0.762
#
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 76.2%
# Adding in Flattening and Shaping after Embedding Layer as suggested by documentation for Embedding.
# Note the range is smaller and thus under states the correct answers and needs to be normalized
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 47, 50)            11700
# _________________________________________________________________
# dropout (Dropout)            (None, 47, 50)            0
# _________________________________________________________________
# flatten (Flatten)            (None, 2350)              0
# _________________________________________________________________
# reshape (Reshape)            (None, 1, 2350)           0
# _________________________________________________________________
# lstm (LSTM)                  (None, 100)               980400
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 101
# =================================================================
# Total params: 992,201
# Trainable params: 992,201
# Non-trainable params: 0
# __________________________
# Accuracy: 100.000000
# Loss: 0.000942
# Min preds: -0.03355398774147034 max_preds: 0.7395308613777161
# predictions  0.0  1.0
# actuals
# 0.0           23    2
# 1.0           14    3
#
# Recall:     0.176
# Precision:  0.600
# Accuracy:   0.619
# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 76.2%
# Added in Flattening and Shaping after Embedding Layer as suggested by documentation for Embedding.
# Normalize the prediction (p - min)/(max - min)
# Accuracy: 100.000000
# Loss: 0.000942
# Min preds: -0.03355398774147034 max_preds: 0.7395308613777161
# predictions  0.0  1.0
# actuals
# 0.0           17    8
# 1.0           10    7
#
# Recall:     0.412
# Precision:  0.467
# Accuracy:   0.571

# ==================================
# Embedding: 50, Dropout: 0.2, LSTM: 100, optmizer: Adam, epochs: 200, Train Tst Split: .3 => Accuracy: 73.8%
# Removed in Flattening and Shaping.
# Normalize the prediction (p - min)/(max - min)
# Accuracy: 100.000000
# Loss: 0.000223
# Min preds: -0.28002065420150757 max_preds: 1.016097068786621
# predictions  0.0  1.0
# actuals
# 0.0           21    4
# 1.0            7   10
#
# Recall:     0.588
# Precision:  0.714
# Accuracy:   0.738
# ==================================
