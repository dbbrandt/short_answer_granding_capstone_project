from source.utils import train_and_test, load_seb_data, evaluate, decode_predictions

def main():
    model_dir = 'model/seb'
    model_file = 'baseline'
    questions_file = f"data/seb/questions.csv"
    train = True
    pretrained = True

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_seb_data(pretrained)

    # For preptained, the vocabulary is some subset of the full Glove dataset. We need the highest key
    # used because the embedding layer assumes a continuous set of values.
    vocab_size = max(vocabulary.keys())+1

    model_params = {'max_answer_len': max_answer_len,
                    'vocab_size': vocab_size,
                    'epochs': 200,
                    'pretrained': pretrained,
                    'embedding_dim': 50,
                    'flatten': False,
                    'lstm_dim_1': 100,
                    'lstm_dim_2': 0,
                    'dropout': 0.2}

    train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test,
                   vocabulary, questions_file, train)

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


# Showing results of test
# _________________________________________________________________
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 48, 50)            11700
# _________________________________________________________________
# dropout (Dropout)            (None, 48, 50)            0
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
# Min preds: -0.24717122316360474 max_preds: 0.9711225032806396
# predictions  0.0  1.0
# actuals
# 0.0           24    1
# 1.0           11    6
#
# Recall:     0.353
# Precision:  0.857
# Accuracy:   0.714
#
# Incorrect Predictions
# id, prediction, correct, answer, correct_answer
# 0, 0.0, "1.0", "the nickel is the hardest becaus is you scratch it with a penni it will not scratch but the nickel could becaus it is so much", "The harder coin will scratch the other."
# 3, 0.0, "1.0", "it is one materi that dissolv into the other make a clear mixtur although it could be color it ha to be see through", "A solution is a mixture formed when a solid dissolves in a liquid."
# 7, 0.0, "1.0", "how you would know that one of them is harder is for say if the nickel never a scratch the penni and the penni got scratch by the nickel then the nickel would be harder than the penni that is how you would which one is harder", "The harder coin will scratch the other."
# 9, 0.0, "1.0", "the one that is harder will scratch the less harder one", "The harder coin will scratch the other."
# 16, 1.0, "0.0", "the ring do not touch becaus the ring are just magnet and the magnet are on the repel side", "Like poles repel and opposite poles attract."
# 18, 0.0, "1.0", "in order to tell which is harder els would have to see if the penni and nickel leav a mark to see which is harder", "The harder coin will scratch the other."
# 20, 0.0, "1.0", "they do not touch each other becaus if you put magnet the magnet like thi will not attract north to north if you put magnet like thi it will attract north to south if you put it like thi it will not attract south to south", "Like poles repel and opposite poles attract."
# 23, 0.0, "1.0", "it would make a high sound", "When the string was tighter, the pitch was higher."
# 25, 0.0, "1.0", "she would scratch the rock togeth the one that scratch is softer", "The harder coin will scratch the other."
# 26, 0.0, "1.0", "it dissolv the solid into a liquid that is see through", "A solution is a mixture formed when a solid dissolves in a liquid."
# 37, 0.0, "1.0", "a solut is when you take a mixtur like powder and mix it with water then after stir it it becom a solut", "A solution is a mixture formed when a solid dissolves in a liquid."
# 40, 0.0, "1.0", "in a solut one of the materi dissolv and the mixtur is see through", "A solution is a mixture formed when a solid dissolves in a liquid."

# Correct Predictions
# id, prediction, correct, answer, correct_answer
# 1, 0.0, "0.0", "becaus a solut is clear but other mixtur are not", "A solution is a mixture formed when a solid dissolves in a liquid."
# 2, 0.0, "0.0", "the magnet do not touch becaus if you flip the magnet to the other side it will not stick becaus it make them differ", "Like poles repel and opposite poles attract."
# 4, 1.0, "1.0", "which had less scratch", "The harder coin will scratch the other."
# 5, 0.0, "0.0", "it is a clear mixtur", "A solution is a mixture formed when a solid dissolves in a liquid."
# 6, 0.0, "0.0", "by use your hand", "The harder coin will scratch the other."
# 8, 0.0, "0.0", "they are not touch each other becaus they repel", "Like poles repel and opposite poles attract."
# 10, 0.0, "0.0", "they are repel", "Like poles repel and opposite poles attract."
# 11, 0.0, "0.0", "the ring do not touch each other becaus if the magnet are not the right way they will repel", "Like poles repel and opposite poles attract."
# 12, 0.0, "0.0", "the magnet are not touch becaus the magnet forc is make the magnet repel", "Like poles repel and opposite poles attract."
# 13, 1.0, "1.0", "when the string is tighten the pitch is higher not lower", "When the string was tighter, the pitch was higher."
# 14, 0.0, "0.0", "when it wa loos the sound wa soft but when it wa tighten it is low", "When the string was tighter, the pitch was higher."
# 15, 1.0, "1.0", "the string wa higher pitch becaus it had more tension", "When the string was tighter, the pitch was higher."
# 17, 0.0, "0.0", "when you make it long it make it low sound when you make it short it a height sound", "When the string was tighter, the pitch was higher."
# 19, 0.0, "0.0", "the nickel is the heavier and the penni is the lighter", "The harder coin will scratch the other."
# 21, 1.0, "1.0", "she would know becaus the one that is scratch is the one that is least hard and the one that is not scratch is the one that is hardest", "The harder coin will scratch the other."
# 22, 0.0, "0.0", "they are on the same side", "Like poles repel and opposite poles attract."
# 24, 0.0, "0.0", "she could tell which is harder by get a rock and see if the penni or nickel would scratch it whichev one doe is harder", "The harder coin will scratch the other."
# 27, 0.0, "0.0", "the magnet are not touch becaus they are flip around a certain way that they do not stick instead they repel", "Like poles repel and opposite poles attract."
# 28, 1.0, "1.0", "the one that is scratch is softer", "The harder coin will scratch the other."
# 29, 0.0, "0.0", "it ha to be a fairli clear", "A solution is a mixture formed when a solid dissolves in a liquid."
# 30, 0.0, "0.0", "the string is lower when you tighten it", "When the string was tighter, the pitch was higher."
# 31, 1.0, "1.0", "the ring do not touch becaus the top is south pole is the one below it is the south pole when the pole are north and south they attract", "Like poles repel and opposite poles attract."
# 32, 0.0, "0.0", "a solut is differ becaus a solut is a mixtur of a solid materi and a liquid", "A solution is a mixture formed when a solid dissolves in a liquid."
# 33, 0.0, "0.0", "mayb he should scratch even other then when you are done see if one have scratch", "The harder coin will scratch the other."
# 34, 0.0, "0.0", "clear and thing that make it still clear", "A solution is a mixture formed when a solid dissolves in a liquid."
# 35, 0.0, "0.0", "i do not know", "The harder coin will scratch the other."
# 36, 0.0, "0.0", "it wa soft then it got louder", "When the string was tighter, the pitch was higher."
# 38, 0.0, "0.0", "becaus of it he you scratch with a penni", "The harder coin will scratch the other."
# 39, 0.0, "0.0", "a solut doe not dissolv", "A solution is a mixture formed when a solid dissolves in a liquid."
# 41, 0.0, "0.0", "the magnet will mayb not stick becaus the forc of the magnet will mayb the pencil is long and the magnet cannot feel the other magnet", "Like poles repel and opposite poles attract."