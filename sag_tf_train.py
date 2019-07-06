from source.utils import train_and_test, load_sag_data

def main():
    model_dir = 'model/sag'
    model_file = '20pct'  # used to load existing model if train is false
    questions_file = 'data/sag2/questions.csv'
    train = True
    # Pretrained embeddings
    pretrained = True
    data_percentage = 1

    X_train, y_train, X_test, y_test, max_answer_len, vocabulary = load_sag_data(pretrained, data_percentage)

    model_params = {'max_answer_len': max_answer_len,
                    'vocab_size': len(vocabulary),
                    'epochs': 30,
                    'pretrained': True,
                    'embedding_dim': 50,
                    'flatten': True,
                    'lstm_dim_1': 100,
                    'lstm_dim_2': 20,
                    'dropout': 0.2}

    # Trains model if train=True and prints out metrics on results (see below)
    train_and_test(model_dir, model_file, model_params, X_train, y_train, X_test, y_test,
                   vocabulary, questions_file, train)

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

# Review Results

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

# Min preds: -0.14248129725456238 max_preds: 1.0054311752319336
# predictions  0.0  1.0
# actuals
# 0.0           98  104
# 1.0           71  460
#
# Recall:     0.866
# Precision:  0.816
# Accuracy:   0.761
#
# Incorrect Predictions
# id, prediction, correct, answer, correct_answer
# 0, 1.0, "0.0", "by column", "By rows."
# 1, 0.0, "1.0", "unlik in a string declar use type string in a string declar use an array of charact the programm must provid the null termin charact and must ensur the array is larg enough to hold the string and null termin charact", "The strings declared using an array of characters have a null element added at the end of the array."
# 6, 1.0, "0.0", "i have an hard time explain thi so i will show how infix is evalu instead start with an infix express like 5 plu 2multipli by 5plus400 divid by2 plu 3 and push item until you get a and onc that happen perform the oper until you reach an with that complet you will now have 7 multipli by 5plus400 divid by2 plu 3a now the express that will be evalu perform last step again 35 plus400 divid by2 plu 3i now the stack repeat agian 35 plus400 divid by 5 is now the stack after that repeat 35 plu 80i now the stack repeat again 115 is now the stack and is return", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 16, 1.0, "0.0", "the differ is that a string that is declar a variabl of type char is each char is includ in the array versu the string of charact that is read into a variabl", "The char [] will automatically add a null 0 character at the end of the string."
# 23, 0.0, "1.0", "declar two variabl front and rear to be use to denot which element in the array can be access increment rear whenev data is enqueu to the end and increment front whenev data is dequeu", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 26, 1.0, "0.0", "basic pop the stack until you find the given element", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 29, 0.0, "1.0", "a doubly-link list allow you to delet a node without travers the list to establish a trail pointer", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 32, 0.0, "1.0", "at the minimum lognor log2 at the maximum n search where n is the number of node", "The height of the treeor log of the number of elements in the tree."
# 39, 1.0, "0.0", "static and dynam", "In the array declaration, or by using an initializer list."
# 40, 1.0, "0.0", "a stack some print job might have a higher prioriti than other and they can be easili insert at the front of the list or anywher between", "queue"
# 44, 1.0, "0.0", "if you use a dynam alloc array you must provid a destructor and copi constructor", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 45, 1.0, "0.0", "a constructor is call whenev a new object of that class is made", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 49, 1.0, "0.0", "there are onli two way to pass someth by valu and by refer use of dot or pointer oper within the receiv function and use of address oper insid the pass function", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 68, 1.0, "0.0", "in array string are store as a characterschar each charact of string will be store in each memori locat of the array while string of charact is onli one memori locat", "The char [] will automatically add a null 0 character at the end of the string."
# 74, 1.0, "0.0", "fifo first in first out", "push"
# 78, 1.0, "0.0", "the signatur can includ a result type and thrown error", "The name of the function and the types of the parameters."
# 86, 0.0, "1.0", "constructor cannot return valu", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 88, 0.0, "1.0", "a string char add a null valu to the end of the string", "The char [] will automatically add a null 0 character at the end of the string."
# 89, 1.0, "0.0", "by implement an array and onli ad item to the end of the array and onli remov item from the begin of the array", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 96, 0.0, "1.0", "all but the first", "All the dimensions, except the first one."
# 104, 1.0, "0.0", "the array itself can be sign to a pointer or each element of the array can be assign to a pointer", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 106, 1.0, "0.0", "the number of node from root to leaf", "The length of the longest path from the root to any of its leaves."
# 107, 1.0, "0.0", "ifright is greater than leftmid equalsright plu leftdivid by 2 m sortnumb temp left mid m sortnumb temp mid plu 1 right mergenumb temp left mid plu 1 right", "When the size of the array to be sorted is 1or 2"
# 109, 0.0, "1.0", "creat a node with the input data continu to add to the list when dequeu get the first element data and set the next element in the list as the new first element", "Keep the rear of the queue pointing to the tail of the linked list, so the enqueue operation is done at the end of the list, and keep the front of the queue pointing to the head of the linked list, so the dequeue operation is done at the beginning of the list."
# 110, 0.0, "1.0", "1 less than the total number of dimens", "All the dimensions, except the first one."
# 115, 0.0, "1.0", "refin the solut", "The testing stage can influence both the coding stagephase 5and the solution refinement stagephase 7"
# 116, 1.0, "0.0", "use link list you are push the node that contain each int to the stack until you get to the end of your link list", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 121, 1.0, "0.0", "you do not alter the origin valu of the variabl that wa pass", "It avoids making copies of large data structures when calling functions."
# 129, 0.0, "1.0", "when you travers a tree of ani size you will visit each node three time it on the order of 3n or onrun time", "A walk around the tree, starting with the root, where each node is seen three times: from the left, from below, from the right."
# 130, 1.0, "0.0", "set the node to null where that it doe not point to anyth and the use the delet opert to clear space from memori", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 136, 1.0, "0.0", "the variabl of type char each charact is store into a differ memori address and can be access easli where as in a string of charact it not easi to be abl to access each charact in the string", "The char [] will automatically add a null 0 character at the end of the string."
# 140, 0.0, "1.0", "n oper where n is the number of item", "Nthe length of the arrayoperations achieved for a sorted array."
# 148, 1.0, "0.0", "first attach the element from the node to be delet to altern node and then delet that node delet node", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 153, 0.0, "1.0", "a static array retain ani modifi valu after a function call automat array reset to their initi valu when the function end", "The static arrays are intialized only once when the function is called."
# 155, 0.0, "1.0", "the role of a header file list all the function a class can do while hide the inner work of it function", "To store a class interface, including data members and member function prototypes."
# 160, 1.0, "0.0", "the bodi of the function that hold all the actual code", "The name of the function and the list of parameters, including their types."
# 161, 1.0, "0.0", "a class definit includ the definit of the class s constructorsand ani public or privat function of cours it also includ the class header and ani necessari c librari", "Function members and data members."
# 170, 1.0, "0.0", "creat an array and implement pointer that point to the next list item down and stack them up", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 174, 1.0, "0.0", "to lay out the basic and give you a start point in the actual problem solv", "To simulate the behaviour of portions of the desired software product."
# 183, 1.0, "0.0", "you store the stack in the array but you have to keep in mind the first element", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 190, 1.0, "0.0", "like a link list except it is first in last out", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 191, 1.0, "0.0", "array it is the collect of similar data type ex int a 10 ten indic the size of array is index of array we can give onli integ valu to array of a where as string mean collect of group of charact string declar have a datatyp usual caus storag to be alloc in memori that is capabl of hold some predetermin number of symbol howev array can be declar to contain valu of ani non refer data type multipl arrari of the same type", "The strings declared using an array of characters have a null element added at the end of the array."
# 199, 1.0, "0.0", "in a doubli link list there are more pointer to set and the mechan of insert and delet are more difficult also the special case at the begin or end of the list are more complic", "Extra space required to store the back pointers."
# 200, 0.0, "1.0", "array item are access directli with equal access time", "The elements in an array can be accessed directlyas opposed to linked lists, which require iterative traversal."
# 201, 1.0, "0.0", "a constructor doe not need a type and it is use to initi the variabl", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 206, 0.0, "1.0", "re-us and eas of mainten", "Abstraction and reusability."
# 207, 0.0, "1.0", "they are transform into post-fix express then evalu with a stack", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 214, 0.0, "1.0", "by search down the tree until you find the node and replac the link to that node with the greatest child node on the left subtre or the least child node on the right subtre", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 217, 1.0, "0.0", "they differenti by the compil by the condit or input use for one of the overload function", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 219, 0.0, "1.0", "to pop element then push them back", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 222, 1.0, "0.0", "class definit includ the name of the class and type of paramet", "Function members and data members."
# 226, 0.0, "1.0", "all string repres by charact array end with the null charact you declar an object of type string just like ani other type for exampl string s", "The strings declared using an array of characters have a null element added at the end of the array."
# 233, 1.0, "0.0", "pick a number and set all valu less than that number to the left while all number on the right of that number is larger", "It selects the minimum from an array and places it on the first position, then it selects the minimum from the rest of the array and places it on the second position, and so forth."
# 238, 1.0, "0.0", "you send a pointer to an object of the linkedlist class", "By reference."
# 241, 1.0, "0.0", "to address major issu in the creation of the program there is no way to account for all possibl bug in the program but it is possibl to prove the program is tangibl", "To simulate the behaviour of portions of the desired software product."
# 244, 0.0, "1.0", "a hierarchi of node that are sort in a particular order each node ha a ancestorexcept for the rootand childrenexcept for the leav", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 257, 0.0, "1.0", "the in order is to go from left root right", "Traverse the left subtree, then the root, then the right subtree."
# 259, 0.0, "1.0", "the advantag is that each node point to both it predecessor and it successor there are no special case for insert and delet", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 262, 0.0, "1.0", "you run a program with differ data size like 10 to the power of x as you increas x and measur the complet speed for the program you can find pattern and attempt the measur the run time it is veri import to keep the same softwar and hardwar howev which make experiment test inferior to theoret in the eye of most", "Implement the algorithm and measure the physical running time."
# 263, 0.0, "1.0", "refin and code", "The testing stage can influence both the coding stagephase 5and the solution refinement stagephase 7"
# 264, 0.0, "1.0", "a binari search tree is a tree that also ha the condit that each node may have at maximum 2 children and where the input data is compar to the data in the tree start with the root if the valu is smaller than the root it travers left if it is larger it travers right until it becom a leaf", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 265, 1.0, "0.0", "it is more effici and it chang the variabl not onlin insid the function but outsid so that the new valu can be use elsewher", "It avoids making copies of large data structures when calling functions."
# 271, 1.0, "0.0", "the function read the variabl store them then return the what ever the variabl read the function then print the content of the array", "by reference."
# 272, 0.0, "1.0", "loglogn2 to the power oflog nn to the power of 2 n to the power of 3 n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 275, 0.0, "1.0", "from least to greatest loglog n 2 to the power oflog n n to the power of 2 n to the power of 3 n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 279, 1.0, "0.0", "by the type they are initil withint char etc", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 283, 0.0, "1.0", "the star return the valu", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 300, 1.0, "0.0", "to get earli feedback from user in earli stage of develop to show user a first idea of what the program will do or look like to make sure the program will meet requir befor intens program begin", "To simulate the behaviour of portions of the desired software product."
# 318, 0.0, "1.0", "store a set of element in a last in first out order", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 322, 0.0, "1.0", "code and refin", "The testing stage can influence both the coding stagephase 5and the solution refinement stagephase 7"
# 324, 1.0, "0.0", "use a link list is one way to implement a stack so that it can handl essenti ani number of element it is usual conveni to put a data structur in it own modul thu you will want to creat file stack h and a stack c", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 325, 1.0, "0.0", "go to the bottom of the left sub tree and visit the parent and then it children", "A walk around the tree, starting with the root, where each node is seen three times: from the left, from below, from the right."
# 327, 1.0, "0.0", "in order start with the root then doe right child then left child recurs", "Traverse the left subtree, then the root, then the right subtree."
# 336, 1.0, "0.0", "if you pass by refer you can modifi the valu as oppos to pass by valu where you cannot chang the valu", "It avoids making copies of large data structures when calling functions."
# 341, 1.0, "0.0", "that depend on the number of data member in the class", "Unlimited number."
# 343, 1.0, "0.0", "data member are the data compon of a particular class a member function are the function compon of the class", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 344, 1.0, "0.0", "global variabl can b access by ani class wit an object in the variabl class", "File scope."
# 347, 1.0, "0.0", "in the test phase", "At the main function."
# 349, 0.0, "1.0", "a constructor set up the default valu of variabl when an object is instanti wherea a function allow interact with that object", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 350, 1.0, "0.0", "the sort array or list is built one entri at a time", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 357, 1.0, "0.0", "no node in a circular link list contain null", "The last element in a circular linked list points to the head of the list."
# 358, 0.0, "1.0", "2 to the power oflog nloglog nn to the power of 2 n to the power of 3 n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 360, 1.0, "0.0", "the intens of the children say you begin with one node that one height then you add two children to that node and then two children to each of those node and two children to each of those node your current height would be 4", "The length of the longest path from the root to any of its leaves."
# 362, 0.0, "1.0", "creat a link list add an element to the top of the stack when push and delet an element when pop", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 367, 1.0, "0.0", "ha preorder in order and postord of a tree preorder equal put the parent node in front of the child node in order equal put the parent node between the left child and right child node postord equal put the parent node after the child node", "A walk around the tree, starting with the root, where each node is seen three times: from the left, from below, from the right."
# 368, 1.0, "0.0", "the declar of the function", "The name of the function and the list of parameters, including their types."
# 373, 1.0, "0.0", "both need to have some kind of base case to tell when the loop need to stop", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 380, 1.0, "0.0", "by argument and refer", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 388, 1.0, "0.0", "you need to pop each item on the stack and compar the item onc the item is found no more item need to be remov", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 402, 1.0, "0.0", "or if you want to initi a variabl to a certain valu equal", "By using constructors."
# 407, 0.0, "1.0", "string is a class in the standard librari and ha method that modifi it wherea the char array is on abl to be modifi by the user", "The strings declared using an array of characters have a null element added at the end of the array."
# 408, 1.0, "0.0", "to provid an exampl or model of how the finish program should perfom provid forsight of some of the challang that would be encount provid opportun to introduc chang to the finish program", "To simulate the behaviour of portions of the desired software product."
# 412, 0.0, "1.0", "they can be declar global just befor the main methodbut also outsid of it or variabl can be subject to onli the method they are scope within but would still be declar at the begin of that methodbut insid of it", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 413, 1.0, "0.0", "a constructor is a function use to initi an object s data when it is creat", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 415, 0.0, "1.0", "to find a node in a binari search tree take at most the same number of step as there are level of the tree", "The height of the tree."
# 418, 0.0, "1.0", "give faster time for some oper such as insert and delet", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 420, 1.0, "0.0", "n multipli by logn", "When the size of the array to be sorted is 1or 2"
# 422, 0.0, "1.0", "by use the multipli bypoint plu element-index", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 423, 1.0, "0.0", "with a number and with a variabl", "In the array declaration, or by using an initializer list."
# 427, 1.0, "0.0", "a constructor is a function that initi a class instanc a function perform whatev task it is written for", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 428, 0.0, "1.0", "string declar use the type string can vari in length string declar use an array of charact can not extend past the array length", "The strings declared using an array of characters have a null element added at the end of the array."
# 431, 0.0, "1.0", "no base case no chang in valu", "If the recursion step is defined incorrectly, or if the base case is not included."
# 439, 1.0, "0.0", "usual it is by row then follow by the column but it is up to the programm to determin how valu are store in bi-dimension array", "By rows."
# 442, 1.0, "0.0", "an object and data", "Data membersattributesand member functions."
# 458, 0.0, "1.0", "depend on the type of stack on a true stack you will haft to pop all of the element of the stack until you find the element you want and then after that you will need to push all the element that where remov back in to the stack in the order that they where remov with a java style stack where there is a peek function it will return the element you want instead of requir you to perform all the excess action that are requir from a true stack", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 459, 1.0, "0.0", "select sort is usual a recurs sort method where you divid the element to be sort in half repeatedli you then sort the smallest case then work your way up sort each until they are all sort", "It selects the minimum from an array and places it on the first position, then it selects the minimum from the rest of the array and places it on the second position, and so forth."
# 460, 1.0, "0.0", "by row and column", "By rows."
# 464, 0.0, "1.0", "stack are a type of contain adaptor specif design to oper in a lifo contextlast-in first-out where element are insert and extract onli from the end of the contain", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 467, 0.0, "1.0", "a static array will be call onli onc onc it ha been declar the complier will ignor the static line if it come to it again", "The static arrays are intialized only once when the function is called."
# 472, 1.0, "0.0", "you need the pop opert to go throught the stack and find the element", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 475, 0.0, "1.0", "in a doubli link list you can delet a node without have to travers the list", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 476, 1.0, "0.0", "thi is shown by the use of cpp and header file by split the program up into smaller subsect of individu code it becom easier to write and keep up with as oppos to have all of your code in one file", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 480, 0.0, "1.0", "to provid inform about the content of a librari it includ the definit of class declar of function data type and constant", "To store a class interface, including data members and member function prototypes."
# 488, 0.0, "1.0", "one of the main advantag is you can hide inner detail a techniqu known by encapsul object combin the data and oper but you cannot see how it work anoth advantag is you can reus class that have been defin earlier in the program a method known as inherit final anoth advantag is object can determin appropri oper at execut time a techniqu known as polymorph", "Abstraction and reusability."
# 492, 0.0, "1.0", "with int x 10 int star of xptr xptr equal x address address of x 1 is the same as xptr plu 1", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 497, 1.0, "0.0", "divid the array in half sort each half then sort them back in one array", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 500, 1.0, "0.0", "it a littl more confus the special case at the end and begin becom more difficult to do", "Extra space required to store the back pointers."
# 501, 0.0, "1.0", "a list-bas implement is prefer becaus the list is more flexibl than a array", "Link-based, because they are dynamicno size constraints"
# 506, 1.0, "0.0", "by referenc a pointer and refer to other pointer rel to the first pointerpoint plu 1 pointer plu 2 etc", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 509, 0.0, "1.0", "when non are provid", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 512, 0.0, "1.0", "a list or array of onli 1 element", "When the size of the array to be sorted is 1or 2"
# 519, 0.0, "1.0", "the size of the list be sent is is less than or equal to 1", "When the size of the array to be sorted is 1or 2"
# 520, 1.0, "0.0", "keep a valu of how mani oper it take and add to thi valu each time a function is call", "Implement the algorithm and measure the physical running time."
# 524, 0.0, "1.0", "write arithmet express is call infix notat thi is becaus a binari operatorlik plusi written in between it two operandsa in a plu b there are two altern form of notat use in certain situat one is prefix notat in which an oper is written befor it operand in prefix notat the sum of a and b is written plu a b thi is the notat use to write function call in mathemat and comput scienc it is also use in the lisp and scheme program languag in postfix notat an oper is written after it operand the sum of a and b is written a b plu you may have seen thi as revers polish notat postfix notat form the conceptu basi for the way that arithmet express are evalu by a comput one import characterist of both postfix and prefix notat is that they are unambigu no parenthes are need to indic the order of oper", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 525, 1.0, "0.0", "a static array cannot be chang in the program", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 528, 1.0, "0.0", "a header file consist of reusabl sourc code such as a class in a file that by convent ha a h filenam extens thi differ from cpp file that contain source-cod", "To store a class interface, including data members and member function prototypes."
# 530, 1.0, "0.0", "to put the biggest element at the end of the list and place the next highest element behind it and so on", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 540, 1.0, "0.0", "it provid a limit proof of concept to verifi with the client befor actual program the whole applic", "To simulate the behaviour of portions of the desired software product."
# 546, 1.0, "0.0", "to sort the element by compar two element and swap the smaller one to sort the element in the array", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 556, 1.0, "0.0", "a tree is a finit set of one or more node such that there is a special design node call the root", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 559, 0.0, "1.0", "you recurs visit everi node on the list you visit the node from the left the bottom and from the right", "A walk around the tree, starting with the root, where each node is seen three times: from the left, from below, from the right."
# 564, 1.0, "0.0", "well for one encapsul the valu of the variabl insid an object are privat unless method are written to pass info outsid of the object as well as inherit where each subclass inherit all variabl and method of it super class exampl in the book includ obj clock and how obj alarm would still use clock from the first class", "Abstraction and reusability."
# 571, 0.0, "1.0", "on the list is alreadi sort", "Nthe length of the arrayoperations achieved for a sorted array."
# 573, 1.0, "0.0", "object are usual initi at the begin of the program and are initi usual in the main function they are initi after the class s name", "By using constructors."
# 576, 0.0, "1.0", "ologn", "The height of the tree."
# 577, 1.0, "0.0", "static and dynam", "In the array declaration, or by using an initializer list."
# 578, 1.0, "0.0", "to show that a certain part of the program work as it is suppos to", "To simulate the behaviour of portions of the desired software product."
# 579, 1.0, "0.0", "specifi an array s size with a constant variabl and set array element with calcul", "In the array declaration, or by using an initializer list."
# 580, 1.0, "0.0", "function prototyp that contain function and data member", "The name of the function and the list of parameters, including their types."
# 587, 1.0, "0.0", "gener a compil error the compil will complain that the variabl you are refer to wa never declar", "Run-time error."
# 588, 1.0, "0.0", "where you onli must merg sort onc", "When the size of the array to be sorted is 1or 2"
# 593, 1.0, "0.0", "an object with a locat in memori where valu can be store", "A location in memory that can store a value."
# 600, 0.0, "1.0", "do not have to make copi of stuff", "It avoids making copies of large data structures when calling functions."
# 606, 1.0, "0.0", "at the root", "At the main function."
# 612, 0.0, "1.0", "log n comparison", "The height of the treeor log of the number of elements in the tree."
# 619, 1.0, "0.0", "you declar your inlin function in the header or befor your int main you then can call that function at anytim in your main program quickli and easili", "It makes a copy of the function code in every place where a function call is made."
# 620, 0.0, "1.0", "compil select proper function to execut base on number type and order of argument in the function call", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 623, 1.0, "0.0", "you can go up and down an array but you can onli go one direct while travers a link list", "The elements in an array can be accessed directlyas opposed to linked lists, which require iterative traversal."
# 626, 0.0, "1.0", "if root set root to null els if delet right leaf set rightptr of parent node to null els if delet left leaf set leftptr of parent node to null els if delet a left or right subtre child node set the max leaf child in the left subtre as the new child node", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 632, 1.0, "0.0", "the role of a prototyp program is to help spot key problem that may aris dure the actual program", "To simulate the behaviour of portions of the desired software product."
# 633, 0.0, "1.0", "compil select proper function to execut base on number type and order of argument in the function call", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 634, 0.0, "1.0", "a basic link list ha an end with a null valu where a circular link list ha a pointer from the end to the begin", "The last element in a circular linked list points to the head of the list."
# 641, 0.0, "1.0", "quickest at top slowest at bottom loglog nn to the power of 2 n to the power of 3 2 to the power oflog nn", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 642, 0.0, "1.0", "a finit set of node that start with the root and termin with leav", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 643, 1.0, "0.0", "variabl can be a integ or a string in a program", "A location in memory that can store a value."
# 646, 1.0, "0.0", "address b 3 is as bptr plu 3 array can be treat as pointer", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 650, 0.0, "1.0", "they can be access by ani part of the program it can be referenc by ani function that follow the declar or definit in the sourc file", "File scope."
# 655, 0.0, "1.0", "array-bas prevent the push oper from ad an item to the stack if the stack s size limit which is the size of the array ha been reach list-bas much simpler to write and doe not have a fix size", "Link-based, because they are dynamicno size constraints"
# 656, 1.0, "0.0", "if they are declar fix or static that mean they cannot chang size onc their storag ha been alloc howev one that is not or dynam arrari can be resiz", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 657, 1.0, "0.0", "a pointer base implement of a queue could use a linear link list with two extern pointer one to the front and one to the back", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 661, 1.0, "0.0", "the euler tour travers of a tree the function that iter keep track of the pointer on each node", "A walk around the tree, starting with the root, where each node is seen three times: from the left, from below, from the right."
# 674, 0.0, "1.0", "boolean isfullqqueu equal rear equal equal max queue size-1", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 677, 1.0, "0.0", "when each half or the origin array ha noth els to sort and put the half back togeth", "When the size of the array to be sorted is 1or 2"
# 678, 1.0, "0.0", "a prototyp program is use in problem solv to collect data for the problem", "To simulate the behaviour of portions of the desired software product."
# 682, 0.0, "1.0", "n minu 1 the best case is when the array is alreadi sort", "Nthe length of the arrayoperations achieved for a sorted array."
# 684, 1.0, "0.0", "one declar as static is one that is alreadi defin the program know the length and the array from the start wherea non-stat array are declar or assign later", "The static arrays are intialized only once when the function is called."
# 686, 1.0, "0.0", "a constructor is a method that start new instanc of a class exampl employe employe 1parametersstart a new instanc of object of type employe a function is simpli a modul within a program that complet it singl desir task", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 689, 1.0, "0.0", "a tree is a data structur where node are link to each other in a hierarch manner", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 692, 1.0, "0.0", "veri simplist it move one element from the list by one and insert them in their correct posit into a new slot then start over best case is01", "Nthe length of the arrayoperations achieved for a sorted array."
# 695, 0.0, "1.0", "it depend if it s a global then they have to be declar out side the sourc code to be use in everi scope howev a local variabl is one declar in a local function etc which obvious doesn t need to be declar outsid the variabl see how it is use for the function or block it be call for", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 699, 1.0, "0.0", "pass by refer with refer argument functionint addresspass by refer with pointer argument functionint star", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 706, 0.0, "1.0", "constructor cannot return valu so they cannot specifi a return type like function can", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 719, 0.0, "1.0", "one or more node in a hierarchi start with the root and branch off like a tree to subtre", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 723, 1.0, "0.0", "a prototyp declar what will be use in the program and the definit", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 731, 1.0, "0.0", "depend on the locat of the node you are look for if it is the root it is one step els if it is smaller than the current you are on node you go to the left if it is larger than the current node you are on go to the right", "The height of the tree."




# Correct Predictions
# id, prediction, correct, answer, correct_answer
# 2, 1.0, "1.0", "static array are avail throughout the program", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 3, 0.0, "0.0", "one", "Unlimited number."
# 4, 1.0, "1.0", "store a set of element in a particular order with a first in first out principl", "A data structure that can store elements, which has the property that the last item added will be the last to be removedor first-in-first-out."
# 5, 1.0, "1.0", "a binari tree where the children are order such that the right side is greater than the current node and the left is less than or equal the current node", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 7, 1.0, "1.0", "the divid and conquer paradigm split a larg problem into simpler problem at which point it solv the simpler problem and merg the simpl solut togeth to answer the larg problem", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 8, 1.0, "1.0", "is a pointer that contain the address of a function", "The address of the location in memory where the function code resides."
# 9, 1.0, "1.0", "member function and data member", "Function members and data members."
# 10, 1.0, "1.0", "the name of the function and the type of it argument thi includ the number type and order the paramet appear in", "The name of the function and the types of the parameters."
# 11, 1.0, "1.0", "when a problem is too big split it into smaller problem of the same type and solv those then from the solut of the smaller problem give the solut to the larger origin problem", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 12, 1.0, "1.0", "a stack is an abstract data type that is base on the principl that the last element insert into the stack will be the first element remov from the stack", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 13, 1.0, "1.0", "variabl that contain the memori address of a data object", "The address of a location in memory."
# 14, 1.0, "1.0", "push and pop", "push and pop"
# 15, 1.0, "1.0", "implicit name when you give it valu like 1 2 3 4 5 6 7 8 9 at which the compil will automat give thi a size static dure the declar", "In the array declaration, or by using an initializer list."
# 17, 1.0, "1.0", "both are base on a control statement both involv repetit both involv a termin test both gradual approach termin both can occur infinit", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 18, 1.0, "1.0", "a travers that visit the left branch first then the parent node then the right branch recurs", "Traverse the left subtree, then the root, then the right subtree."
# 19, 1.0, "1.0", "the base case is when the length of the current array is 1", "When the size of the array to be sorted is 1or 2"
# 20, 1.0, "1.0", "the function signatur tell what the function paramet and includ all the function call", "The name of the function and the list of parameters, including their types."
# 21, 1.0, "1.0", "when a constructor is not provid by the programm of the class", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 22, 1.0, "1.0", "all dimens exclud the first one", "All the dimensions, except the first one."
# 24, 1.0, "1.0", "by their function signatur", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 25, 1.0, "1.0", "main", "At the main function."
# 27, 1.0, "1.0", "more memori can be alloc on the fli for more item", "The linked lists can be of variable length."
# 28, 1.0, "1.0", "push and pop", "push and pop"
# 30, 0.0, "0.0", "binari search tree are similar to binari tree but have been implement to organ data in a specif way for later search", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 31, 1.0, "1.0", "the sequenc of number ha zero or one element", "When the size of the array to be sorted is 1or 2"
# 33, 1.0, "1.0", "becaus you cannot chang the origin and pass by refer limit memori need for the program", "It avoids making copies of large data structures when calling functions."
# 34, 1.0, "1.0", "the element typic includ in a class definit are the function prototyp usual declar public and the data member use in the class which are usual declar privat", "Function members and data members."
# 35, 0.0, "0.0", "the inord travers of a binari tree visit the node of a binari tree in order accord to their data valu", "Traverse the left subtree, then the root, then the right subtree."
# 36, 0.0, "0.0", "in the veri begin of the program befor the main start", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 37, 1.0, "1.0", "the prototyp creat a framework to call the function definit while a function definit is where the function is actual program out and creat into a final product", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 38, 1.0, "1.0", "a function definit doe not requir ani addit inform that need to be pass insid it parenthesi to execut while a definit prototyp requir more than one paramet to be pass in order to complet it task", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 41, 1.0, "1.0", "pop which remov the head or least recent insert node from the stack push which insert a new node at the head of the stack", "push and pop"
# 42, 1.0, "1.0", "thi oper return the memori address of it operand", "The memory address of its operand."
# 43, 1.0, "1.0", "util a front pointer and a back pointer the front poiner and back pointer point to the first item into the queue as you add item the front remain the same but the back poiner next point to the new item and the new item is assign to becom the new back pointer", "Keep the rear of the queue pointing to the tail of the linked list, so the enqueue operation is done at the end of the list, and keep the front of the queue pointing to the head of the linked list, so the dequeue operation is done at the beginning of the list."
# 46, 1.0, "1.0", "a binari tree is a tree data structur in which each node ha at most two children", "A tree for which the maximum number of children per node is two."
# 47, 1.0, "1.0", "infix express are convert to postfix i e 3 plu 2 is chang to 32 plu", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 48, 1.0, "1.0", "lognwher n is the number of node", "The height of the tree."
# 50, 1.0, "1.0", "the height of a tree is how mani level of node that it ha", "The length of the longest path from the root to any of its leaves."
# 51, 1.0, "1.0", "use some sort of count principl of the number of oper perform in an algorithm", "Implement the algorithm and measure the physical running time."
# 52, 0.0, "0.0", "it depend what type of class is be defin typic you would have a constructor call for each object", "Unlimited number."
# 53, 1.0, "1.0", "the delet of a node depend upon if it ha children and if it is an avl binari search tree assum it is not an avl tree and the node be delet ha no children you just set it pointer to null if it ha a left child or a right child exclus that child replac the delet node if it ha two children the left most child of the right sub treeor right most child of the left subtreewil replac it", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 54, 1.0, "1.0", "enqueu dequeu", "enqueue and dequeue"
# 55, 1.0, "1.0", "log n", "The height of the tree."
# 56, 1.0, "1.0", "no easili reach base case and no base case at all", "If the recursion step is defined incorrectly, or if the base case is not included."
# 57, 1.0, "1.0", "privat public protect", "Private and public."
# 58, 1.0, "1.0", "the comput convert the infix express to postfix form then evalu the postfix express", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 59, 0.0, "0.0", "they can travers in both direct", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 60, 1.0, "1.0", "run-tim", "Run-time error."
# 61, 1.0, "1.0", "run-tim error", "Run-time error."
# 62, 1.0, "1.0", "variabl that exist throught the entir script their valu can be chang anytim in the code and fucntion", "File scope."
# 63, 1.0, "1.0", "the last child on ani branch", "A node that has no children."
# 64, 1.0, "1.0", "a binari search tree is a tree that also ha the condit that each node may have at maximum 2 children", "A tree for which the maximum number of children per node is two."
# 65, 1.0, "1.0", "by refer", "by reference."
# 66, 1.0, "1.0", "a queue is a storag contain that hold it object in a first in first out prioriti", "A data structure that can store elements, which has the property that the last item added will be the last to be removedor first-in-first-out."
# 67, 0.0, "0.0", "it creat a set of candid function then a set of viabl function", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 69, 1.0, "1.0", "link list is abl to grow in size as need doe not requir the shift of item dure insert and delet", "The linked lists can be of variable length."
# 70, 1.0, "1.0", "a function prototyp is a declar of a function that tell the compil the function s name it return type and the type of it paramet", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 71, 1.0, "1.0", "you search the tree for the node use recurs when you find the node you determin whether it is a leaf or a intern node if it is a leaf you just delet it and set the parent pointer to that node to null if it is a node you replac the node with either of the children node", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 72, 0.0, "0.0", "object are initi by give the object a type name and initi valu", "By using constructors."
# 73, 1.0, "1.0", "the size is not fix easier sort becaus of no shift easier to insert item into the list", "The linked lists can be of variable length."
# 75, 0.0, "0.0", "exist class can be reus and program mainten and verif are easier", "Abstraction and reusability."
# 76, 1.0, "1.0", "it expand the function s definit in that place onc the function is call you can use it as mani time as you need the compil just expand on the function", "It makes a copy of the function code in every place where a function call is made."
# 77, 1.0, "1.0", "it divid the problem into singular unit and work on the problem piec by piec until the problem is solv", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 79, 1.0, "1.0", "a pointer is a variabl that hold the address of a given variableand of a given data type", "The address of a location in memory."
# 80, 1.0, "1.0", "a pointer is a refer to a memori locat", "The address of a location in memory."
# 81, 1.0, "1.0", "all element are initi to zero if not explicitli initi thi doe not happen for automat local array", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 82, 1.0, "1.0", "merg sort continu break an array in half then sort the array as it concaten them back togeth into one sort array", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 83, 1.0, "1.0", "it organ data in a nonlinear hierarch form where item can have more than one successor partit into a root node and subset are gener subtre of the root", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 84, 1.0, "1.0", "by the number and type of argument", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 85, 1.0, "1.0", "the star oper return the object at that memori locat", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 87, 1.0, "1.0", "includ the name accept paramet and return type", "The name of the function and the list of parameters, including their types."
# 90, 1.0, "1.0", "push", "push"
# 91, 1.0, "1.0", "a while statement will test the condit of the while loop first there is a chanc the loop will never run a do while loop will alway run onc and then the while test will determin if it will run again", "The block inside a do ... while statement will execute at least once."
# 92, 1.0, "1.0", "queue", "queue"
# 93, 1.0, "1.0", "link list onli allow sequenti access where array allow random access", "The elements in an array can be accessed directlyas opposed to linked lists, which require iterative traversal."
# 94, 0.0, "0.0", "by be pass to the function by a lead term", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 95, 1.0, "1.0", "logn", "The height of the tree."
# 97, 1.0, "1.0", "a method with access to a link list s head pointer as access to the entir list", "By reference."
# 98, 1.0, "1.0", "an array is capabl of access ani part of that array base on the index the link list must be travers from the begin or the end that is data can onli be access if it is adjac to the previou or next node", "The elements in an array can be accessed directlyas opposed to linked lists, which require iterative traversal."
# 99, 1.0, "1.0", "look at the 2nd element move forward and place the element in the correct spot", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 100, 1.0, "1.0", "use node to keep track of the head of the stack then use push and pop to creat the stack as need", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 101, 1.0, "1.0", "a queue it would not be unfair for the first job to finish last", "queue"
# 102, 1.0, "1.0", "start at the begin of an array take each element in order and place it in it is correct posit rel to all previous sort element", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 103, 0.0, "0.0", "initi use the same name as the class", "By using constructors."
# 105, 1.0, "1.0", "push and pop", "push and pop"
# 108, 0.0, "0.0", "not answer", "Extra space required to store the back pointers."
# 111, 1.0, "1.0", "an array ha a fix size you can add and delet element to the end of the array and you use a pointer to keep track of the last element ad each time you add or delet an element you updat the pointer and check if it is equal to the max size of the array", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 112, 1.0, "1.0", "global variabl have program scopeaccess anywher in program", "File scope."
# 113, 1.0, "1.0", "to sort the element in an array by remov an element from the input data and insert it at the correct posit", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 114, 1.0, "1.0", "in a circular link list the last element point to the first", "The last element in a circular linked list points to the head of the list."
# 117, 1.0, "1.0", "push which add an element to the stack and pop which take an element off the stack", "push and pop"
# 118, 0.0, "0.0", "on", "Nthe length of the arrayoperations achieved for a sorted array."
# 119, 1.0, "1.0", "a data structur that contain a root intern node and extern node each node refer anoth node by mean of pointerspass-by-refer the root is the base of the tree it ha no parent a leaf is a node at the end of the tree which point to null", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 120, 1.0, "1.0", "by row of row", "By rows."
# 122, 1.0, "1.0", "if the recurs function never reach or success defin the base case it will recurs forev thi happen mani way such as the function doe not progress toward the base case or the function is code poorli and doe not even contain a base case", "If the recursion step is defined incorrectly, or if the base case is not included."
# 123, 0.0, "0.0", "by use an array of charact you are limit to the size of the array of charact by declar by type the end of the string is acknowledg by white space", "The strings declared using an array of characters have a null element added at the end of the array."
# 124, 1.0, "1.0", "return synonym for the object it operand point to", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 125, 1.0, "1.0", "infinit recurs is an infinit loop if the condit is not met either omit the base case or write the recurs step incorrectli so that it doe not converg on the base case caus indefinit recurs eventu exhaust memori", "If the recursion step is defined incorrectly, or if the base case is not included."
# 126, 1.0, "1.0", "a stack is similar to an array but doe not allow for random access stack onli allow a user to retriev the last item put into the stack last in fist out", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 127, 1.0, "1.0", "array are pass by refer", "by reference."
# 128, 1.0, "1.0", "a function prototyp tell the compil the function name return type and the number and type of paramet without reveal the implement contain in the function definit", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 131, 1.0, "1.0", "push and pop", "push and pop"
# 132, 1.0, "1.0", "enqueu and dequeu", "enqueue and dequeue"
# 133, 1.0, "1.0", "a function signatur use in a function s prototyp is the set of object type it take in as paramet with or without name given for the object", "The name of the function and the list of parameters, including their types."
# 134, 1.0, "1.0", "use a pointer that alway point to the end of the array list for push or pop modif", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 135, 1.0, "1.0", "is the number of gener in the tree", "The length of the longest path from the root to any of its leaves."
# 137, 1.0, "1.0", "manual insid the bracket or automat via an initi list", "In the array declaration, or by using an initializer list."
# 138, 1.0, "1.0", "uniqu function signatur", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 139, 0.0, "0.0", "compil error", "Run-time error."
# 141, 1.0, "1.0", "pointer is a program data type whose valu point to anoth valu store in comput memori by it address", "The address of a location in memory."
# 142, 1.0, "1.0", "the array of charact ha a null charact 0 at the end of the array to signifi the array s end the string doe not have thi", "The strings declared using an array of characters have a null element added at the end of the array."
# 143, 1.0, "1.0", "local variabl can onli be use within the function where as data member can be set to public access and can be use throughout", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 144, 1.0, "1.0", "the base case for a recurs implement of merg sort is when the sequenc be pass to merg sort ha less than 2 element", "When the size of the array to be sorted is 1or 2"
# 145, 1.0, "1.0", "a data structur for store item which are to be access in last-in first-out order that can be implement in three way", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 146, 1.0, "1.0", "the address oper return the memori address of the variabl it preced", "The memory address of its operand."
# 147, 1.0, "1.0", "data member are variabl that are declar insid the class definit but outsid of the bodi of the class member function local variabl can onli be use within the function declar", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 149, 1.0, "1.0", "a function that solv a problem by divid the problem into smaller problem by call it self again and again until a base case is reach", "A function that calls itself."
# 150, 1.0, "1.0", "use iter loop", "Through iteration."
# 151, 1.0, "1.0", "they can be use throughout the program", "File scope."
# 152, 1.0, "1.0", "at the main function", "At the main function."
# 154, 1.0, "1.0", "use the head as the top of the stack onli modifi the head when you push or pop push would add a new item to the head pop would remov the item from the head", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 156, 1.0, "1.0", "a circular link list doe not have a last element instead it is last item point to the head of the list", "The last element in a circular linked list points to the head of the list."
# 157, 1.0, "1.0", "nonconst pointer to nonconst data nonconst pointer to constant data constant pointer to nonconst data and constant pointer to constant data", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 158, 1.0, "1.0", "a type of data structur in which each element is attach to one or more element directli beneath it", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 159, 1.0, "1.0", "push add an element to the top of the stack pop remov the top element from the stack", "push and pop"
# 162, 1.0, "1.0", "an array can be address in pointer or offset notat by set a pointer variabl equal to the variabl name of the array element of the array can then be access by ad an offset valu to the pointer variabl", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 163, 1.0, "1.0", "minu a function with access to a link list s head pointer ha access to the entir list pass the head ponter to a function as a refer argument", "By reference."
# 164, 1.0, "1.0", "doe not have a fix size link list is abl to grow as need the time to access an array base list take a contant amount of time where as an linked-bas like depend on i", "The linked lists can be of variable length."
# 165, 0.0, "0.0", "header file have reusabl sourc code in a file that a program can use", "To store a class interface, including data members and member function prototypes."
# 166, 1.0, "1.0", "insid the function scope and outsid of the function scope in case of global variabl", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 167, 0.0, "0.0", "they are basic the same howev a string end with a null charact denot the end of the stringand the size a char array ha potenti to be ani size so it must be declar or limit", "The strings declared using an array of characters have a null element added at the end of the array."
# 168, 1.0, "1.0", "sizeof return the size in byte of the respect object", "The size in bytes of its operand."
# 169, 0.0, "0.0", "not answer", "Run-time error."
# 171, 1.0, "1.0", "main function", "At the main function."
# 172, 1.0, "1.0", "an endpoint on a tree that contain no pointer or pointer that are set to null", "A node that has no children."
# 173, 1.0, "1.0", "a queue is a list of data that follow the fifo principl an exampl of thi would be when you get into a line at a movi theater the first one there get to buy a ticket first", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 175, 0.0, "0.0", "the main role of header file is it is use to share inform among variou file", "To store a class interface, including data members and member function prototypes."
# 176, 1.0, "1.0", "pop an element from one stack check to see if it is the desir element if not push it onto anoth stack when finish pop the item from the second stack and push them back onto the first stackthi will ensur the order of the element is maintain", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 177, 1.0, "1.0", "m-by-n by row-column", "By rows."
# 178, 1.0, "1.0", "public and privat specifi", "Private and public."
# 179, 0.0, "0.0", "elabor construct and transit are all affect by test", "The testing stage can influence both the coding stagephase 5and the solution refinement stagephase 7"
# 180, 1.0, "1.0", "insert sort divid the list into sort and unsort region then take each item from the unsort region and insert it into it correct order in the sort region", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 181, 1.0, "1.0", "it contain the address of the function in memori", "The address of the location in memory where the function code resides."
# 182, 1.0, "1.0", "if it ha no children you just delet it if it onli ha one child just replac the node with whichev child it ha if it ha both children replac it with one of it children and send the other child down along the other side of the new node", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 184, 0.0, "0.0", "not answer", "Implement the algorithm and measure the physical running time."
# 185, 1.0, "1.0", "after load the requir includ statement and librari the main method begin the execut", "At the main function."
# 186, 0.0, "0.0", "by use an array of charact one can store and manipul the string rather than just have a type string variabl", "The strings declared using an array of characters have a null element added at the end of the array."
# 187, 0.0, "0.0", "static long unsign", "Private and public."
# 188, 1.0, "1.0", "a local variabl cannot be access outsid the function in which it is declar data member normal are privat variabl of function declar privat are access onli to member function of the class in which they are declar", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 189, 1.0, "1.0", "dequeu and enqueu", "enqueue and dequeue"
# 192, 1.0, "1.0", "overal the program ha better performancemean it is fasterbecaus it doe not have to copi larg amount of data", "It avoids making copies of large data structures when calling functions."
# 193, 1.0, "1.0", "array are pass to function by refer", "by reference."
# 194, 1.0, "1.0", "you do not have to iter through the entir list to access element", "The elements in an array can be accessed directlyas opposed to linked lists, which require iterative traversal."
# 195, 1.0, "1.0", "a data type that contain a pointer to at least the next element in a list", "A collection of elements that can be allocated dynamically."
# 196, 0.0, "0.0", "you would have to travers the stack pop each element to search it", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 197, 1.0, "1.0", "a node with degre 0 last node in the tree and furtherest away from the root", "A node that has no children."
# 198, 1.0, "1.0", "an array that is declar as static will retain the valu store in it is element between function call and will not reiniti them to default valu", "The static arrays are intialized only once when the function is called."
# 202, 1.0, "1.0", "point to the memori address of a function kind of like break a branch off of a tree object and hit other object with it", "The address of the location in memory where the function code resides."
# 203, 1.0, "1.0", "multipl copi of the function code are insert into the program make it bigger", "It makes a copy of the function code in every place where a function call is made."
# 204, 1.0, "1.0", "print a tree in order from least to greatest thi done by go as far left down the tree as possibl and print the parent and then right tree then move up the tree", "Traverse the left subtree, then the root, then the right subtree."
# 205, 1.0, "1.0", "in a binari search tree you must first establish a proper replac for the node you are about to delet usual a child from the soon to be delet node onc that replac node ha been found you simpli reassign it to where the node that is go to be delet is after the delet node ha been usurp you remov the delet node from memori so it may be use again", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 208, 1.0, "1.0", "data member and function definit", "Function members and data members."
# 209, 1.0, "1.0", "list-bas becaus it is not fix size", "Link-based, because they are dynamicno size constraints"
# 210, 0.0, "0.0", "onlogn", "When the size of the array to be sorted is 1or 2"
# 211, 1.0, "1.0", "a local variabl is onli useabl within the function it is defin wherea a data member is avail to ani method within it class", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 212, 0.0, "0.0", "all of the dimens must be specifi", "All the dimensions, except the first one."
# 213, 0.0, "0.0", "either travers the entir list and pop the given part or creat a pointer system that automat point to it", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 215, 0.0, "0.0", "to hide the definit and detail of a class also to help readabl of the main c plu plu file", "To store a class interface, including data members and member function prototypes."
# 216, 0.0, "0.0", "variabl are usual declar at the begin of a modul of c plu plu code", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 218, 1.0, "1.0", "nearli infinit size limit onli by system memori and also the abil to expand the size dynam", "The linked lists can be of variable length."
# 220, 1.0, "1.0", "divid element recur then conquer which work in merg sort and quicksort", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 221, 1.0, "1.0", "the compil distinguish overload function by their signatur it encod each function identifi with the number and type of it paramet to gener type-saf linkag which ensur the proper overload function is call", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 223, 1.0, "1.0", "not have the proper case to leav the recurs", "If the recursion step is defined incorrectly, or if the base case is not included."
# 224, 0.0, "0.0", "linear logarithm exponenti linear linear", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 225, 1.0, "1.0", "the function s name and paramet", "The name of the function and the types of the parameters."
# 227, 1.0, "1.0", "the address is a unari oper that return the memori address of it operand", "The memory address of its operand."
# 228, 1.0, "1.0", "use link list and stack you would need a temp stack to retain the valu then you would use the pop function to pop off each element and then compar it if it not the element your look for push it to the temp stack repeat until the element is found when you find it pop off the temp stack back onto the regular stack to have a complet stack again", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 229, 1.0, "1.0", "push and pop", "push and pop"
# 230, 1.0, "1.0", "take a problem and divid it into a smaller problem and solv that smaller problem or divid it into a smaller problem and solv it thu solv the whole problem in the process", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 231, 1.0, "1.0", "modular reusabl code allow faster deploy of solut and a more gener view of a solut", "Abstraction and reusability."
# 232, 1.0, "1.0", "run-tim error", "Run-time error."
# 234, 1.0, "1.0", "how to determin the end of the list in basic link list the last element link to a null pointer while circular link list link to the head element at the end", "The last element in a circular linked list points to the head of the list."
# 235, 0.0, "0.0", "no answer", "By rows."
# 236, 1.0, "1.0", "a first in first out list of item like if you put 5 4 3 2 and 1 in the queue it will when you dequeu item remov the item in the same order as put in so thu it will put out 5 4 3 2 and 1 in that exact order", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 237, 1.0, "1.0", "a binari tree where the search key in ani node n is greater than the search key in ani node in n s left subtre but less than the search key in ani node in n s right subtre", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 239, 1.0, "1.0", "public can be access from outsid the class privat access onli from insid the class not inherit protect access onli from insid the class inherit", "Private and public."
# 240, 1.0, "1.0", "by-dimension array are store by row", "By rows."
# 242, 1.0, "1.0", "make an array of a size and add on to the front and delet from the back keep track of the two so that you know when it is full and where to add or subtract from", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 243, 0.0, "0.0", "a static array can be can be edit throughout the program while a non-stat array can onli be edit within a given function", "The static arrays are intialized only once when the function is called."
# 245, 1.0, "1.0", "list-bas becaus memori is not constrict", "Link-based, because they are dynamicno size constraints"
# 246, 0.0, "0.0", "compil error", "Run-time error."
# 247, 1.0, "1.0", "link list do not have a set size and can grow or shrink as need", "The linked lists can be of variable length."
# 248, 1.0, "1.0", "increas memori requir slightli more complic when modifi element in the list", "Extra space required to store the back pointers."
# 249, 1.0, "1.0", "split the problem into smaller more manag part and proceed to address the smaller problem", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 250, 1.0, "1.0", "nonconst pointer to nonconst data nonconst pointer to constant data constant pointer to nonconst data constant pointer to constant data", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 251, 0.0, "0.0", "abil to backtrack through a list", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 252, 1.0, "1.0", "a pointer is a variabl that contain a memori address for someth that you can use such as a valu array or even a function", "The address of a location in memory."
# 253, 1.0, "1.0", "a pointer to a function itself contain the address of the function and can be use to call that function", "The address of the location in memory where the function code resides."
# 254, 1.0, "1.0", "a link list with a first in out structur dequeu at the head of the list enqueu at the end of the list", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 255, 1.0, "1.0", "a while loop is pre-checkit check the condit statement befor it execut the code within the while blocka do while loop is post-checkit check the condit after the block execut it run at least onc no matter what the condit statement is", "The block inside a do ... while statement will execute at least once."
# 256, 1.0, "1.0", "the variabl of type string ha a termin charact 0 at the end of it", "The char [] will automatically add a null 0 character at the end of the string."
# 258, 1.0, "1.0", "a variabl is the memori address for a specif type of store data or from a mathemat perspect a symbol repres a fix definit with chang valu", "A location in memory that can store a value."
# 260, 1.0, "1.0", "the local variabl is lost onc it exit the block of code while the data member is not", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 261, 0.0, "0.0", "the advantag is that oop allow us to build class of object three principl that make up oop are encapsulation-object combin data and oper inheritance-class can inherit properti from other class polymorphism-object can determin appropri oper at execut time", "Abstraction and reusability."
# 266, 1.0, "1.0", "row", "By rows."
# 267, 1.0, "1.0", "a function pointer is a pointer that contain the address of the function in memori", "The address of the location in memory where the function code resides."
# 268, 1.0, "1.0", "defin in the specif phase a prototyp stimul the behavior of portion of the desir softwar product mean the role of a prototyp is a temporari solut until the program itself is refin to be use extens in problem solv", "To simulate the behaviour of portions of the desired software product."
# 269, 1.0, "1.0", "they are pointer that contain the address to function they can be pass and return from function as well as store in array and assign to other function pointer", "The address of the location in memory where the function code resides."
# 270, 1.0, "1.0", "the number of gener or level of a tree", "The length of the longest path from the root to any of its leaves."
# 273, 0.0, "0.0", "give call function the abil to access and modifi the caller s argument data directli", "It avoids making copies of large data structures when calling functions."
# 274, 1.0, "1.0", "pointer that contain the address of function", "The address of the location in memory where the function code resides."
# 276, 1.0, "1.0", "you would go to the furthest down left most node then to the root then to the rightif left and right existthen you would return one node previou and do the same until you reach the root then go to the furthest down left most node on the right side of the root and continu thi process", "Traverse the left subtree, then the root, then the right subtree."
# 277, 1.0, "1.0", "a data structur that put element in a list and onli allow the user access to the last element", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 278, 1.0, "1.0", "the compil differenti overload function by their signatur", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 280, 1.0, "1.0", "tell the compil to make a copi of function s code in place to avoid a function call it typic ignor it except for the smallest function", "It makes a copy of the function code in every place where a function call is made."
# 281, 0.0, "0.0", "the compil can ignor the inlin qualifi and typic doe so for all but the smallest function", "It makes a copy of the function code in every place where a function call is made."
# 282, 1.0, "1.0", "link list are abl to grow in size so element can be ad to the list", "Linked lists are dynamic structures, which allow for a variable number of elements to be stored."
# 284, 1.0, "1.0", "the byte size of the date store ina variabl", "The size in bytes of its operand."
# 285, 1.0, "1.0", "a recurs function onli know how to solv base case a recurs function call itself directli or indirectli until a base case is reach", "A function that calls itself."
# 286, 1.0, "1.0", "public privat and protect", "Private and public."
# 287, 1.0, "1.0", "store by row", "By rows."
# 288, 1.0, "1.0", "if the node ha no children delet it right away otherwis put either the furthest right node on the left side or the furthest left node on the right side in that place and perform a the abov on that node to guarante that it is children get handl properli", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 289, 1.0, "1.0", "insert sort is were after k iter the first k item in the array are sort it take the k plu 1 item and insert it into the correct posit in the alreadi sort k element", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 290, 1.0, "1.0", "data member class variabl and function", "Function members and data members."
# 291, 1.0, "1.0", "push which insert an element on the top of the stack and pop which remov the last insert element from the stack", "push and pop"
# 292, 1.0, "1.0", "a queue store a set of element in a particular order it principl of oper is fifofirst in first out which mean the first element insert is the first one to be remov", "A data structure that can store elements, which has the property that the last item added will be the last to be removedor first-in-first-out."
# 293, 1.0, "1.0", "by have the head pointer point to the first or least current data enter and have the tail point to the most current data enter a method must be creat so that the tail pointer doe not leav the array", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 294, 1.0, "1.0", "it a locat in memori that contain the memori address of anoth locat in memori that contain inform", "A variable that contains the address in memory of another variable."
# 295, 1.0, "1.0", "static use and dynam use", "In the array declaration, or by using an initializer list."
# 296, 1.0, "1.0", "a constructor cannot return valu it not even void it is use to initi an object s data when it is creat wherea a function is creat to do a specif task and it can return valu", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 297, 1.0, "1.0", "the complier includ copi of inlin function instead of make function call but usual onli with veri small function", "It makes a copy of the function code in every place where a function call is made."
# 298, 0.0, "0.0", "2 to the power oflog n n to the power of 3 n to the power of 2 loglog n n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 299, 1.0, "1.0", "log n", "The height of the tree."
# 301, 1.0, "1.0", "merg sort use the divid and conquer idea where it divid the array in half multipl time and then join each element of the array back into one sort array thi is one of the best sort algorithm besid quicksort", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 302, 1.0, "1.0", "array of charact need a termin charact as well as size specif whether it explicit or implicit", "The strings declared using an array of characters have a null element added at the end of the array."
# 303, 1.0, "1.0", "the portion of a function prototyp that includ the name of the function and the type of it argument", "The name of the function and the types of the parameters."
# 304, 0.0, "0.0", "you can pass them with the pointerstaror the memori addressaddress", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 305, 1.0, "1.0", "break a singl array down into mani array with individu element then sort the element as you reconstruct them back into a singl array", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 306, 0.0, "0.0", "postord", "Traverse the left subtree, then the root, then the right subtree."
# 307, 1.0, "1.0", "take an array and split it into two then solv these simpler problem and merg the two answer in correct order", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 308, 1.0, "1.0", "to simul problem solv for part of the problem", "To simulate the behaviour of portions of the desired software product."
# 309, 0.0, "0.0", "you can travers the list in revers", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 310, 1.0, "1.0", "at most it equival to the height of the tree", "The height of the treeor log of the number of elements in the tree."
# 311, 1.0, "1.0", "use an iter call", "Through iteration."
# 312, 1.0, "1.0", "array are good for random access and good for sequenti access which are both in constant time where link list are linear for random access array are faster in thi case", "The elements in an array can be accessed directlyas opposed to linked lists, which require iterative traversal."
# 313, 1.0, "1.0", "a link list is a data structur contain one or more data element with a pointer to the next node", "A collection of elements that can be allocated dynamically."
# 314, 1.0, "1.0", "the memori address of it operand", "The memory address of its operand."
# 315, 1.0, "1.0", "a leaf is a child of a parent node that ha no children node of it own", "A node that has no children."
# 316, 0.0, "0.0", "charact array will termin at ani whitespac includ space string termin when they encount the new line charact", "The char [] will automatically add a null 0 character at the end of the string."
# 317, 1.0, "1.0", "a set of node that is either empti or partit into a root node and one or two subset that are binari subtre of the root each node ha at most two children the left child and the right child", "A tree for which the maximum number of children per node is two."
# 319, 1.0, "1.0", "a link list is a data structur that is not necessarili in the same contigu memori spacesuch as array it hold the data type and point to the next data item in the list or in a doubli link list also to the previou item", "A collection of elements that can be allocated dynamically."
# 320, 1.0, "1.0", "by refrenc", "by reference."
# 321, 1.0, "1.0", "return synonym for the object it operand point to", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 323, 1.0, "1.0", "a comparison sort in which the sort array is built one entri at a time", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 326, 0.0, "0.0", "you can travers the list both forward and backward", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 328, 0.0, "0.0", "3 step at most there are 3 case", "The height of the tree."
# 329, 1.0, "1.0", "push", "push"
# 330, 0.0, "0.0", "anyth you can do recurs you can do iter", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 331, 1.0, "1.0", "push and pop", "push and pop"
# 332, 0.0, "0.0", "give function abil to access and modifi the caller s argument data directli", "It avoids making copies of large data structures when calling functions."
# 333, 1.0, "1.0", "insert sort progress through a list of element and determin where the next element should be insert into an alreadi sort array start with sort and use the first two element", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 334, 1.0, "1.0", "run the input with variou input measur the run time with system time", "Implement the algorithm and measure the physical running time."
# 335, 1.0, "1.0", "select sort find the lowest element in the data set and place it behind the pivot point", "It selects the minimum from an array and places it on the first position, then it selects the minimum from the rest of the array and places it on the second position, and so forth."
# 337, 1.0, "1.0", "a locat in memori where valu can be store", "A location in memory that can store a value."
# 338, 1.0, "1.0", "when no user-defin constructor exist", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 339, 0.0, "0.0", "function can directli modifi argument that are pass by refer", "It avoids making copies of large data structures when calling functions."
# 340, 1.0, "1.0", "a constructor privat and public variabl and function prototyp", "Function members and data members."
# 342, 1.0, "1.0", "a c plu plu program will begin to execut at the main function", "At the main function."
# 345, 1.0, "1.0", "public and privat", "Private and public."
# 346, 0.0, "0.0", "you would need to perform a search through the list of elementsi dont realli understand what thi question is ask it not veri clear", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 348, 1.0, "1.0", "a data structur that store data use lifo", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 351, 1.0, "1.0", "return the address number of the specifi variabl", "The memory address of its operand."
# 352, 1.0, "1.0", "the altern method is to use loop in the program instead of a function which call itself", "Through iteration."
# 353, 1.0, "1.0", "function name and input paramat", "The name of the function and the types of the parameters."
# 354, 1.0, "1.0", "an object that point to a specif place in memori where a variabl or valu is store", "The address of a location in memory."
# 355, 1.0, "1.0", "a link list ha a dynam size but an array onli ha a fix size and take allot of extra oper to increas it size", "Linked lists are dynamic structures, which allow for a variable number of elements to be stored."
# 356, 1.0, "1.0", "run the code for n-time and get averag valu drop the constant and lowest number for exampl if fxequal 3n plu 1 the run time will bef fxequal on", "Implement the algorithm and measure the physical running time."
# 359, 1.0, "1.0", "a function that call itself in order to solv a problem", "A function that calls itself."
# 361, 1.0, "1.0", "a function signatur is the return type of a function it name and the number and type of it paramet", "The name of the function and the list of parameters, including their types."
# 363, 1.0, "1.0", "the altern way to solv a problem that could be solv use recurs is iter", "Through iteration."
# 364, 1.0, "1.0", "a program intial static local array when their declar are first encount if a static array is not initi explicityli by the programm earch element of that array is intial to zero by the compil when the array is creat non-stat array member cannot be initi at all in c plu plu", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 365, 1.0, "1.0", "the total number of byte of an object", "The size in bytes of its operand."
# 366, 0.0, "0.0", "refin product and mainten", "The testing stage can influence both the coding stagephase 5and the solution refinement stagephase 7"
# 369, 0.0, "0.0", "easier to debugg reusabl", "Abstraction and reusability."
# 370, 0.0, "0.0", "object orient program allow programm to use an object with class that can be chang and manipul while not affect the entir object at onc the class all hold attrubut that affect the object", "Abstraction and reusability."
# 371, 1.0, "1.0", "pass the head pointer to a function as a refer argument", "By reference."
# 372, 0.0, "0.0", "no", "To store a class interface, including data members and member function prototypes."
# 374, 1.0, "1.0", "base on control statement involv repetit involv a termin test both can occur infinitli", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 375, 1.0, "1.0", "by pass the head pointer and go through the list as need insid the function", "By reference."
# 376, 1.0, "1.0", "when the class doe not explicitli includ a constructor", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 377, 1.0, "1.0", "a queue is a data structur where the first node in is the first node out", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 378, 1.0, "1.0", "you do not use unessesari memori space to copi variabl between function", "It avoids making copies of large data structures when calling functions."
# 379, 1.0, "1.0", "a recurs function is a function that call itself repeatedli until a base case is achiev the fundament idea is to break one larg problem into a seri of smaller similar problem", "A function that calls itself."
# 381, 0.0, "0.0", "minu 1 divid by 2", "The height of the tree."
# 382, 1.0, "1.0", "a node with no children", "A node that has no children."
# 383, 1.0, "1.0", "both are base on a control statement iteration-repetit structur recursion-select structur both involv repetit iteration-explicitli use repetit structur recursion-rep function call both involv a termin test iteration-loop-termin test recursion-bas case both gradual approach termin iteration-modifi counter until loop-termin test fail recursion-produc progress simpler version of problem both can occur indefinit iteration-if loop-continu condit never fail recursion-if recurs step doe not simplifi the problem", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 384, 1.0, "1.0", "a singl element on the array", "When the size of the array to be sorted is 1or 2"
# 385, 1.0, "1.0", "left child impli parent impli right child", "Traverse the left subtree, then the root, then the right subtree."
# 386, 1.0, "1.0", "convert infix express to postfix express and evalu the postfix express", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 387, 1.0, "1.0", "a recurs function is a function that call itself usual call the base case if the base case is not correct it caus a infinit loop", "A function that calls itself."
# 389, 1.0, "1.0", "all element are initi to zero if not explicitli initi for a static array while a non-stat array is not initi to zero", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 390, 1.0, "1.0", "it examin the name type and order of argument on each function", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 391, 1.0, "1.0", "a divide-and-conqu paradigm take some data divid it into two part and work on each part indiviu until the item is found", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 392, 1.0, "1.0", "name of the function and the type of it is argument", "The name of the function and the types of the parameters."
# 393, 1.0, "1.0", "find the smallest and put it in the current posit till you get to the end", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 394, 0.0, "0.0", "none just pass the array name", "All the dimensions, except the first one."
# 395, 0.0, "0.0", "not answer", "By using constructors."
# 396, 1.0, "1.0", "a list of object that follow the rule first in first out essenti a link list that goe in order of the first object in the list is the first to be taken off", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 397, 1.0, "1.0", "a link list is one of the fundament data structur and can be use to implement other data structur it consist of a sequenc of node each contain arbitrari data field and one or two referenceslinkspoint to the next and previou node", "A collection of elements that can be allocated dynamically."
# 398, 0.0, "0.0", "instruct the compil on how to interfac with librari and user-written compon", "To store a class interface, including data members and member function prototypes."
# 399, 1.0, "1.0", "comput usual convert infix express to post fix express and evalu them use a stack", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 400, 1.0, "1.0", "as mani as you want as long as they each have a uniqu argument list", "Unlimited number."
# 401, 0.0, "0.0", "remov the element then shift the element one space back", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 403, 0.0, "0.0", "string with type string are just that string they are not part of an array list at all where as one declar by an array is actual an array of charact abl to be point and detect", "The strings declared using an array of characters have a null element added at the end of the array."
# 404, 0.0, "0.0", "class definit are place here", "To store a class interface, including data members and member function prototypes."
# 405, 1.0, "1.0", "it gener a copi of the function code in place to avoid a function call", "It makes a copy of the function code in every place where a function call is made."
# 406, 1.0, "1.0", "explicitli by declar it in bracketsi e int array 50 and implicitli by initi sever valuesi e int array equal 1 2 3", "In the array declaration, or by using an initializer list."
# 409, 1.0, "1.0", "it ha no base case or the base case is never met", "If the recursion step is defined incorrectly, or if the base case is not included."
# 410, 1.0, "1.0", "loglog n 2 to the power oflog n n to the power of 2 n to the power of 3 n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 411, 1.0, "1.0", "if the array length is less than or equal to 1 then that array is return to the other array and merg togeth", "When the size of the array to be sorted is 1or 2"
# 414, 1.0, "1.0", "oposit of a theoret assess of the algorithm to determin runtim but to run the code first to determin the the runtim thi is not recommend becaus it is a limit test it doe not includ all possibl of the data nor the hardwar use to process the data", "Implement the algorithm and measure the physical running time."
# 416, 1.0, "1.0", "there can be infinit constructor as long as the signatur is differ", "Unlimited number."
# 417, 1.0, "1.0", "the array of charact ha a set length while the type string ha virtual unlimit length", "The strings declared using an array of characters have a null element added at the end of the array."
# 419, 0.0, "0.0", "an array is pass by refer therefor if an array of charact is chang the memori is chang not just the variabl", "The strings declared using an array of characters have a null element added at the end of the array."
# 421, 1.0, "1.0", "when you do not provid your own constructor", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 424, 0.0, "0.0", "the size of the string", "The size in bytes of its operand."
# 425, 1.0, "1.0", "list becaus it size is not defin", "Link-based, because they are dynamicno size constraints"
# 426, 0.0, "0.0", "the type string is a class and is safer while the other is just an array of charact", "The strings declared using an array of characters have a null element added at the end of the array."
# 429, 1.0, "1.0", "pop and push", "push and pop"
# 430, 1.0, "1.0", "make a link list and add on to the front and delet from the back keep track of both to do so", "Keep the rear of the queue pointing to the tail of the linked list, so the enqueue operation is done at the end of the list, and keep the front of the queue pointing to the head of the linked list, so the dequeue operation is done at the beginning of the list."
# 432, 1.0, "1.0", "a function that call itself in the definit code", "A function that calls itself."
# 433, 1.0, "1.0", "the divid and conquer paradigm divid a problem into smaller and smaller portion that are easier to solv", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 434, 1.0, "1.0", "in a circular link list the last element point to the head of the list", "The last element in a circular linked list points to the head of the list."
# 435, 1.0, "1.0", "select sort search the array for the lowest valu and swap it with the first valu in the array then search for the next lowest valu and swap it with the second item in the array and so on", "It selects the minimum from an array and places it on the first position, then it selects the minimum from the rest of the array and places it on the second position, and so forth."
# 436, 1.0, "1.0", "merg sort break the array in half and continu to do so until it ha 2 element to compar and sort them after do so it merg back as it keep on sort the algorithm as it doe so", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 437, 1.0, "1.0", "a constructor is automat call whenev an instanc of a class is creat a function must be explicitli call by the user", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 438, 1.0, "1.0", "a leaf is a node with no children", "A node that has no children."
# 440, 1.0, "1.0", "by row", "By rows."
# 441, 1.0, "1.0", "specifi the number of element in the array declar with a constant or use a constant variabl for futur scalabl", "In the array declaration, or by using an initializer list."
# 443, 1.0, "1.0", "a list size of 1 where it is alreadi sort", "When the size of the array to be sorted is 1or 2"
# 444, 1.0, "1.0", "it a run-tim error", "Run-time error."
# 445, 1.0, "1.0", "a class is an expand concept of a data structur it hold both the data and the function be execut", "Data membersattributesand member functions."
# 446, 0.0, "0.0", "it includ the specif inform about the function such as input and output variabl type and how mani of each", "The name of the function and the types of the parameters."
# 447, 1.0, "1.0", "place the smallest item in the list at posit 1 and then proce to each valu until the last posit of the ray is reach", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 448, 0.0, "0.0", "pointer to the child and delet it ha 2 children set the node to the child and delet it the node to th middl will then take it place", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 449, 1.0, "1.0", "enqueu dequeu", "enqueue and dequeue"
# 450, 1.0, "1.0", "by default just one but they may be overload to creat as mani constructor as necessari", "Unlimited number."
# 451, 1.0, "1.0", "no node in a circular link list contain null the last node point pack to a node within the list", "The last element in a circular linked list points to the head of the list."
# 452, 1.0, "1.0", "privat public protect or friend", "Private and public."
# 453, 1.0, "1.0", "list base solut are prefer becaus they allow for queue of ani size", "Link-based, because they are dynamicno size constraints"
# 454, 1.0, "1.0", "a list of object where each object contain a link to the next item in the list", "A collection of elements that can be allocated dynamically."
# 455, 0.0, "0.0", "from lowest to longest loglog n 2 to the power oflog n n n to the power of 2 n to the power of 3", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 456, 1.0, "1.0", "by refer", "by reference."
# 457, 1.0, "1.0", "list base is prefer becaus it is not constrain to a fix size the down fall is that it take up more memori becaus each node ha at least two part the item and the pointer", "Link-based, because they are dynamicno size constraints"
# 461, 1.0, "1.0", "constructor is a special block of statement call when an object is creat either when it is declar static or construct on the stack howev a function is a portion of code within a larger program which perform a specif task and independ to the rest of the code", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 462, 1.0, "1.0", "it use the divid and conqur techniqu recurs and then when merg back togeth it compar each element togeth in a sort list thi is done by revers the divid and conquer techniqu", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 463, 0.0, "0.0", "2 to the power of n where n is the of level the binari tree ha", "The height of the tree."
# 465, 1.0, "1.0", "throught the programm", "File scope."
# 466, 1.0, "1.0", "it goe through the list onli onc pick each integ and put it in it desir posit then continu", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 468, 1.0, "1.0", "in the main function", "At the main function."
# 469, 1.0, "1.0", "a pointer to a function is the address where the code for the function resid they can be pass to function return from function store in array and assign to other pointer", "The address of the location in memory where the function code resides."
# 470, 1.0, "1.0", "thi is a run-tim error or execution-tim error", "Run-time error."
# 471, 1.0, "1.0", "at the main function int main", "At the main function."
# 473, 1.0, "1.0", "it return the size of an array in byte", "The size in bytes of its operand."
# 474, 1.0, "1.0", "a constructor is a function use to initi an object s data when it is creat it is call is made implicitli when the object is creat and must be defin with the same name as the class constructor also cannot return a valu like a function can", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 477, 0.0, "0.0", "string that are use in an char array are much easier to manipul than as a string object becaus each charact is store separ rather than as a whole", "The strings declared using an array of characters have a null element added at the end of the array."
# 478, 1.0, "1.0", "pop push", "push and pop"
# 479, 0.0, "0.0", "binari search tree are a fundament data structur use to construct more abstract data structur such as set multiset and associ array", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 481, 1.0, "1.0", "a merg sort recurs divid the array into half until onli one element remain then it sort the data on it way out of the recurs call by merg the cell", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 482, 1.0, "1.0", "a queue", "queue"
# 483, 1.0, "1.0", "ad a new item and remov the item", "push and pop"
# 484, 1.0, "1.0", "a binari tree where the valu in ani node n is greater than the valu in everi node in n s left subtre but less than the valu of everi node in n s right subtre", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 485, 1.0, "1.0", "the address of the variabl it is attach to", "The memory address of its operand."
# 486, 1.0, "1.0", "you can use an initi list or simpli tell the compil how mani element you want in the array for an initi list int a equal 10 2 3 4 5 for an element declar int b 5 both array have 5 element in them but array a is alreadi initi", "In the array declaration, or by using an initializer list."
# 487, 1.0, "1.0", "the array can act as a pointer or be referenc by a pointer multipli byarrayptr plu 3or multipli byarray plu 3", "By initializing a pointer to point to the first element of the array, and then incrementing this pointer with the index of the array element."
# 489, 1.0, "1.0", "a function signatur should includ the name of the function paramet and a bodi", "The name of the function and the list of parameters, including their types."
# 490, 1.0, "1.0", "a variabl that store the address of a memori locat", "The address of a location in memory."
# 491, 1.0, "1.0", "it elimit the need to copi larg amont of data", "It avoids making copies of large data structures when calling functions."
# 493, 1.0, "1.0", "creat a fix array size with with 2 integ to point to the begin and the end of the que and special case to know when the que is empti or full", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 494, 1.0, "1.0", "the data can be modifi directli instead of make a copi of the data improv execut time with larg amount of data", "It avoids making copies of large data structures when calling functions."
# 495, 1.0, "1.0", "a function that call itself to perform a certain task", "A function that calls itself."
# 496, 1.0, "1.0", "loglog n n to the power of 2 n to the power of 3 2 to the power oflog n n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 498, 0.0, "0.0", "header file contain code which can be use in multipl file", "To store a class interface, including data members and member function prototypes."
# 499, 1.0, "1.0", "public privat protect", "Private and public."
# 502, 1.0, "1.0", "use a link list with 2 pointer one to the front and one to the back as long as back equal front the queue is not empti", "Keep the rear of the queue pointing to the tail of the linked list, so the enqueue operation is done at the end of the list, and keep the front of the queue pointing to the head of the linked list, so the dequeue operation is done at the beginning of the list."
# 503, 1.0, "1.0", "a link list is a chain of node that each store a singl piec of data and pointer variabl that point to other node in the list", "A collection of elements that can be allocated dynamically."
# 504, 1.0, "1.0", "all subsequ dimens after the first dimens first is not need to be specifi", "All the dimensions, except the first one."
# 505, 1.0, "1.0", "a static local array exist for the durat of the program and it element are initi to 0 if not explicitli initi so a static local array s element will still be the same when call later unless specif initi to someth els thi doe not happen for automat array", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 507, 1.0, "1.0", "pointer variabl that point to function address", "The address of the location in memory where the function code resides."
# 508, 1.0, "1.0", "alloc an array of some size bottom stack element store at element 0", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 510, 0.0, "0.0", "variabl are set to a given valu or 0 if none is given", "By using constructors."
# 511, 1.0, "1.0", "it select the smallest element in a list and switch it with the element in it correct posit then it select the next smallest and doe the same", "It selects the minimum from an array and places it on the first position, then it selects the minimum from the rest of the array and places it on the second position, and so forth."
# 513, 1.0, "1.0", "in a circular link list the the last item point to the first item", "The last element in a circular linked list points to the head of the list."
# 514, 1.0, "1.0", "they are both base on a control statement both involv repetit both involv a termin case both graduatlli approach that termin case and both can occur infinit", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 515, 1.0, "1.0", "enqueu and dequeu", "enqueue and dequeue"
# 516, 0.0, "0.0", "the compil can ignor the inlin qualifi and typic doe so for all but the smallest function", "It makes a copy of the function code in every place where a function call is made."
# 517, 1.0, "1.0", "char array need an end charact signatur and is made up of charact each separ from each other a string is an object in itself with a valu that the user enter", "The char [] will automatically add a null 0 character at the end of the string."
# 518, 1.0, "1.0", "a queue is a data structur that store element in a first in first out order", "A data structure that can store elements, which has the property that the last item added will be the last to be removedor first-in-first-out."
# 521, 1.0, "1.0", "by the number type and order of their argument", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 522, 1.0, "1.0", "a pointer variabl contain the number of a memori address as it valu which may be null or 0 or the address of some valu store in memori", "A variable that contains the address in memory of another variable."
# 523, 1.0, "1.0", "larg data item can be pass without copi the entir data point reduc execut time and the amout of memori space need", "It avoids making copies of large data structures when calling functions."
# 526, 1.0, "1.0", "you alloc an pre-defin array the bottom element is store at element 0 and the last index is the head", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 527, 1.0, "1.0", "no base case or an incorrectli written recurs step that doe not converg on the base case will lead to infinit recurs", "If the recursion step is defined incorrectly, or if the base case is not included."
# 529, 0.0, "0.0", "it contain reusabl sourc code for use by other class", "To store a class interface, including data members and member function prototypes."
# 531, 1.0, "1.0", "is equal to the number of level level between the root node and the termin node", "The length of the longest path from the root to any of its leaves."
# 532, 1.0, "1.0", "a function that call itself until it reach a base case", "A function that calls itself."
# 533, 1.0, "1.0", "push and pop", "push and pop"
# 534, 1.0, "1.0", "function prototyp describ the class s public interfac without reveal the class s member function implement function definit show what implement are be done", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 535, 1.0, "1.0", "find the smallest valu in the list and make it the first element then find the smallest valu of the leftov list and make it the first element of the leftov list and continu until the list is sort", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 536, 1.0, "1.0", "run-tim", "Run-time error."
# 537, 1.0, "1.0", "the valu of the object that the operand point to", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 538, 0.0, "0.0", "an array declar as static can onli be declar onc", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 539, 1.0, "1.0", "a while loop check if the condit is true or not first if it wa true it excut the statement a do while loop execut the statement befor it check the condit if the condit wa true it would excut the statement again so a do while loop would excut the statement atleast onc", "The block inside a do ... while statement will execute at least once."
# 541, 1.0, "1.0", "list base it can dynam grow and ha fewer restrict", "Link-based, because they are dynamicno size constraints"
# 542, 1.0, "1.0", "push and pop", "push and pop"
# 543, 1.0, "1.0", "push equal enqueu pop equal dequeu", "push"
# 544, 1.0, "1.0", "you could use the first element of the list as the remov point and insert at the end f you do thi you would have to shift the element down each time you remov an item unless you make the array circular", "Use a circular array. Keep the rear of the queue toward the end of the array, and the front toward the beginning, and allow the rear pointer to wrap around."
# 545, 1.0, "1.0", "an object list that store element in a particular order the first object insert is at the bottom with the last object at the top so the first object in is the last object out", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 547, 1.0, "1.0", "a public and privat area that includ the function and variabl that are use in the class", "Data membersattributesand member functions."
# 548, 1.0, "1.0", "the number of gener or level the tree ha", "The length of the longest path from the root to any of its leaves."
# 549, 0.0, "0.0", "objectnam classnam to call a function from the class objectnam funciton", "By using constructors."
# 550, 1.0, "1.0", "mergesort divid the array into smaller halv and then combin the sort subarray into one sort array", "It splits the original array into two, sorts each of the two halves, and then merges the sorted arrays."
# 551, 1.0, "1.0", "iter", "Through iteration."
# 552, 1.0, "1.0", "run-tim error", "Run-time error."
# 553, 0.0, "0.0", "an array of string read the string liter mean anyth within quot a char array read a string as each individu charact", "The char [] will automatically add a null 0 character at the end of the string."
# 554, 1.0, "1.0", "infinit recurs may occur if no base case is defin or if the call is not vari", "If the recursion step is defined incorrectly, or if the base case is not included."
# 555, 1.0, "1.0", "the entir program", "File scope."
# 557, 1.0, "1.0", "the compil will provid one when class doe not explictli includ a consructor", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 558, 1.0, "1.0", "onli constant can be use to declar the size of automat and static array exist for the durat of the program is initi when it declar is first encount all element are initi to zero if not explicitli initi", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 560, 1.0, "1.0", "defin an array and keep track of a pointer to the last element as item are ad", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 561, 1.0, "1.0", "a special binari tree that ha a rule that all the subtre on the right are smaller than the node valu and all the subtre on the left are larger than the node valu", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 562, 1.0, "1.0", "in postfix format", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 563, 1.0, "1.0", "an array declar as static is not creat and and initi each time the function and it is also not destroy when the function termin", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 565, 1.0, "1.0", "class variabl class function prototyp", "Function members and data members."
# 566, 1.0, "1.0", "a while statement will onli process if the statement is met while a do while will alway process onc then onli continu if the statement is met", "The block inside a do ... while statement will execute at least once."
# 567, 1.0, "1.0", "a string contain a null charact at the end of the string which make it easili possibl to get the string length a char array can have a virtual unlimit length therefor it size must be declar or limit", "The char [] will automatically add a null 0 character at the end of the string."
# 568, 1.0, "1.0", "the inlin keyword advis the compil to copi the function s code in place to avoid function call howev the compil can and typic doe ignor the inlin qualifi for all but the smallest function", "It makes a copy of the function code in every place where a function call is made."
# 569, 1.0, "1.0", "enquedata dequ", "enqueue and dequeue"
# 570, 1.0, "1.0", "run time error", "Run-time error."
# 572, 0.0, "0.0", "the primari disadvantag of doubli link list are that1each node requir an extra pointer requir more space and2th insert or delet of a node take a bit longermor pointer oper", "All the deletion and insertion operations can be performed in constant time, including those operations performed before a given location in the list or at the end of the list."
# 574, 1.0, "1.0", "break up veri larg data structur into smaller sub-unit that are easier to manipul", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 575, 1.0, "1.0", "pop and push to remov an element and to insert an element", "push and pop"
# 581, 0.0, "0.0", "just one per class", "Unlimited number."
# 582, 0.0, "0.0", "a static array ha a predetermin size and that size cannot be alter", "The static arrays are intialized only once when the function is called."
# 583, 0.0, "0.0", "a group of data in a parent to child structur", "A collection of nodes, which has a special node called root, and the rest of the nodes are partitioned into one or more disjoint sets, each set being a tree."
# 584, 1.0, "1.0", "pass the head pointer of a link list to a function give that function access to all node of that link list", "By reference."
# 585, 1.0, "1.0", "list base becuas an arrari base ha to have an arrari size and need to be pre-defin and cannot be chang dynam", "Link-based, because they are dynamicno size constraints"
# 586, 0.0, "0.0", "elabor construct and transit", "The testing stage can influence both the coding stagephase 5and the solution refinement stagephase 7"
# 589, 1.0, "1.0", "a function that call itself each time it doe it must get smaller and eventu must converg to a base case otherwis you can start an infinit loop", "A function that calls itself."
# 590, 1.0, "1.0", "push", "push"
# 591, 1.0, "1.0", "ani problem solv recurs could be solv with an iter function iter replac for recurs function may be more difficult to program but often lead to more effici solut to a problem", "Through iteration."
# 592, 1.0, "1.0", "within the main function", "At the main function."
# 594, 0.0, "0.0", "on", "Nthe length of the arrayoperations achieved for a sorted array."
# 595, 1.0, "1.0", "a tree which is split base on valu thi make it veri easi to search one can compar the desir valu to the root and if the root is greater than we search the left side of the tree if it is less than we search the right side and do the same thing recurs", "A binary tree that has the property that for any node the left child is smaller than the parent which in turn is smaller than the right child."
# 596, 1.0, "1.0", "consist of sequenc of node each contain a number of data field and one or two link call pointer that point to the next or previou node", "A collection of elements that can be allocated dynamically."
# 597, 0.0, "0.0", "a constructor initi an object or object of a class a function of a class perform a task such as display a line of text or do some kind of mathemat oper", "A constructor is called whenever an object is created, whereas a function needs to be called explicitely. Constructors do not have return type, but functions have to indicate a return type."
# 598, 1.0, "1.0", "the last valu in the tree a valu with no children attatch", "A node that has no children."
# 599, 0.0, "0.0", "ad up the number of oper perform base on the worst case possibl", "Implement the algorithm and measure the physical running time."
# 601, 1.0, "1.0", "return the size in byte of the specifi data", "The size in bytes of its operand."
# 602, 1.0, "1.0", "the idea of divid and conquer is to take a larg problem split it into n smaller problem make the program easier to read and modifi", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 603, 1.0, "1.0", "return synonym for the object it operand point to", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 604, 1.0, "1.0", "for inlin function the compil creat a copi of the function s code in place so it doe not have to make a function call and add to the function call stack", "It makes a copy of the function code in every place where a function call is made."
# 605, 1.0, "1.0", "the valu of the variabl the pointer point to", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 607, 1.0, "1.0", "function pointer are pointer i e variabl which point to the address of a function", "The address of the location in memory where the function code resides."
# 608, 1.0, "1.0", "a queue is a linear first-in first-out data structur data must be access in the same order it wa put into the queue so onli the oldest item in the queue is access at ani time main function defin are enqueu and dequeu", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 609, 1.0, "1.0", "queue", "queue"
# 610, 1.0, "1.0", "return a synonym for the object to which it pointer operand point", "An aliassynonymfor the name of the object that its operand points to in memory. It is the dereferencing operator."
# 611, 1.0, "1.0", "global variabl are declar in the main function local variabl are declar in ani other function", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 613, 1.0, "1.0", "push put an element on the stack pop-tak an element off the stack", "push and pop"
# 614, 1.0, "1.0", "the number of node on the longest path from the root of the tree to a leaf", "The length of the longest path from the root to any of its leaves."
# 615, 1.0, "1.0", "loglogn 2 to the power of lognn to the power of 2 n to the power of 3 n", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 616, 1.0, "1.0", "the size of operand in byte", "The size in bytes of its operand."
# 617, 1.0, "1.0", "a queue is a data structur that hold a set of object which ha a fifofirst in first outprior", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 618, 1.0, "1.0", "it return the memori address of it is operand that is if appli to a normal variabl it give the variabl s memori address just as a pointer variabl might", "The memory address of its operand."
# 621, 1.0, "1.0", "an array of charact ha one element a string doesnt have it is the termin element or null", "The strings declared using an array of characters have a null element added at the end of the array."
# 622, 1.0, "1.0", "a link list is not fix in size and doe not requir the shift of item dure insert and delet", "The linked lists can be of variable length."
# 624, 1.0, "1.0", "a c plu plu class definit may includ access-specifi public privat and static", "Private and public."
# 625, 1.0, "1.0", "a do while statement will alway execut the do piec of code at least onc befor check the condit a while statement will alway check the condit first", "The block inside a do ... while statement will execute at least once."
# 627, 1.0, "1.0", "push and pop are the two main function of a stack", "push and pop"
# 628, 1.0, "1.0", "by alloc an array of predetermin size and an integ to track the top element of the stack the bottom member of the stack will go in element 0 of the array and for each element push the top track integ is increment", "Keep the top of the stack toward the end of the array, so the push and pop operations will add or remove elements from the right side of the array."
# 629, 1.0, "1.0", "lack of defin a base case or write the recurs step incorrectli so that it doe not converg on the base case", "If the recursion step is defined incorrectly, or if the base case is not included."
# 630, 1.0, "1.0", "in a circular link list the last node point to the first node", "The last element in a circular linked list points to the head of the list."
# 631, 0.0, "0.0", "not answer", "Implement the algorithm and measure the physical running time."
# 635, 1.0, "1.0", "1 declar the length of the arrayint array 10 2 initi the arrayint array equal 0 1 2 3 or or compil will assum size of 4", "In the array declaration, or by using an initializer list."
# 636, 1.0, "1.0", "run-tim error", "Run-time error."
# 637, 1.0, "1.0", "in the main function", "At the main function."
# 638, 1.0, "1.0", "a pointer to a locat in memori", "A location in memory that can store a value."
# 639, 1.0, "1.0", "do while statement evalu whether or not to loop after run the block contain within it at least onc so the main differ is that while statement have a possibl of never be use do while statement on the other hand are alway run at least onc befor evalu whether to run again", "The block inside a do ... while statement will execute at least once."
# 640, 1.0, "1.0", "public privat protect", "Private and public."
# 644, 1.0, "1.0", "it is a locat in the comput s memori where it can be store for use by a program", "A location in memory that can store a value."
# 645, 1.0, "1.0", "iter thi would be more effici and ha repetit structur", "Through iteration."
# 647, 0.0, "0.0", "n minu 1", "Nthe length of the arrayoperations achieved for a sorted array."
# 648, 1.0, "1.0", "use head as the top and push and pop node from the head", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 649, 1.0, "1.0", "it replac all instanc of that function call with the inlin code itself result in longer but faster program", "It makes a copy of the function code in every place where a function call is made."
# 651, 1.0, "1.0", "it look at the number type and order of argument in the function call", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 652, 1.0, "1.0", "it includ a function name and paramet list doe not includ return type function signatur must be differ", "The name of the function and the types of the parameters."
# 653, 0.0, "0.0", "you cannot delet a node becaus that can caus a node to have more than 2 children", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 654, 0.0, "0.0", "when no paramet are set is when a default constructor is use", "If no constructor is provided, the compiler provides one by default. If a constructor is defined for a class, the compiler does not create a default constructor."
# 658, 1.0, "1.0", "mani recurs solut may also be solv with loop control statement such as while for do-whil etc", "Through iteration."
# 659, 1.0, "1.0", "the node with degre 0", "A node that has no children."
# 660, 1.0, "1.0", "element are onli insert and remov from the head of the list there is no header node or current pointer", "Keep the top of the stack pointing to the head of the linked list, so the push and pop operations will add or remove elements at the beginning of the list."
# 662, 1.0, "1.0", "it return type and it input paramet", "The name of the function and the list of parameters, including their types."
# 663, 1.0, "1.0", "queue", "queue"
# 664, 1.0, "1.0", "suppli an integ insid the bracket or the compil count the number of element in the initi list int n 5 int n equal 1 2 3 4 5", "In the array declaration, or by using an initializer list."
# 665, 1.0, "1.0", "two paramat the array and how mani column arraya 3", "All the dimensions, except the first one."
# 666, 1.0, "1.0", "list it dynam and no size need to be declar", "Link-based, because they are dynamicno size constraints"
# 667, 1.0, "1.0", "queue would be prefer to stack for use as schedul print job becaus it would print job in the order that they were sent to the printer", "queue"
# 668, 1.0, "1.0", "it start with the second element and check it to see if it is less than the elementsto the left of it and if it is it insert it into it correct posit", "Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left."
# 669, 1.0, "1.0", "comput convert infix express to postfix form befor evalu", "First, they are converted into postfix form, followed by an evaluation of the postfix expression."
# 670, 1.0, "1.0", "a list in which onli the topor last item addedcan be modifi", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 671, 1.0, "1.0", "a function definit is the code that defin the function place in the bracket that determin that function s oper a function prototyp show the function s public interfac without expos implement it show name return type and type of paramat", "A function prototype includes the function signature, i. e., the name of the function, the return type, and the parameters 's type. The function definition includes the actual body of the function."
# 672, 0.0, "0.0", "the list-bas implement is prefer sinc the big o1i veri effici", "Link-based, because they are dynamicno size constraints"
# 673, 1.0, "1.0", "list-bas it can grow and shrink dynam unlik the fix size array", "Link-based, because they are dynamicno size constraints"
# 675, 0.0, "0.0", "pointer may be pass to function by valu by refer with refer argument or by refer with pointer argument", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 676, 1.0, "1.0", "a function that call itself until the base case are met", "A function that calls itself."
# 679, 0.0, "0.0", "it is use to let the user have a first idea of the complet program and allow the client to evalu the program thi can gener much feedback includ softwar specif and project estim of the total project", "To simulate the behaviour of portions of the desired software product."
# 680, 1.0, "1.0", "there is no predetermin length", "Linked lists are dynamic structures, which allow for a variable number of elements to be stored."
# 681, 1.0, "1.0", "the statement within the block of the do while loop will alway be execut at least onc regardless of the condit wherea the while loop may never be execut if the condit are not met", "The block inside a do ... while statement will execute at least once."
# 683, 1.0, "1.0", "push pop", "push and pop"
# 685, 0.0, "0.0", "it treat them as the same function", "It makes a copy of the function code in every place where a function call is made."
# 687, 1.0, "1.0", "n oper the best case scenario is when all the number are in increas order", "Nthe length of the arrayoperations achieved for a sorted array."
# 688, 1.0, "1.0", "a data structur that move element in last in first out", "A data structure that can store elements, which has the property that the last item added will be the first to be removedor last-in-first-out"
# 690, 1.0, "1.0", "contain the address of the function in memori", "The address of the location in memory where the function code resides."
# 691, 1.0, "1.0", "if the function is small enough it will expand it but it will run faster as it will avoid make so mani call to the function", "It makes a copy of the function code in every place where a function call is made."
# 693, 1.0, "1.0", "it a lot like a stack except that the first item put into the list is the first item to be taken from the list", "A data structure that stores elements following the first in first out principle. The main operations in a queue are enqueue and dequeue."
# 694, 1.0, "1.0", "in a circular link list the last node point back to the first node there is no null", "The last element in a circular linked list points to the head of the list."
# 696, 0.0, "0.0", "not answer", "The arrays declared as static live throughout the life of the program; that is, they are initialized only once, when the function that declares the array it is first called."
# 697, 1.0, "1.0", "the height of a tree is the number of node on the longest path from the root to a leaf", "The length of the longest path from the root to any of its leaves."
# 698, 0.0, "0.0", "pass-by-valu or pass-by-refer", "There are four ways: nonconstant pointer to constant data, nonconstant pointer to nonconstant data, constant pointer to constant data, constant pointer to nonconstant data."
# 700, 1.0, "1.0", "iter and recurs both use repetit and perform sever simpl oper and algorithm success until they reach a certain limit so both involv a termin test to find that limit and both slowli approach that termin limit both are base on a control statement as well if code poorli both can continu on for forev until the compil or the comput either lock up shut down or halt the oper", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 701, 1.0, "1.0", "both iter and recurs are base on control statement and involv repetit they can both also occur indefinit", "They both involve repetition; they both have termination tests; they can both occur infinitely."
# 702, 1.0, "1.0", "pointer are variabl that contain as their valu memori address of other variabl", "A variable that contains the address in memory of another variable."
# 703, 0.0, "0.0", "as a pointer node", "By reference."
# 704, 1.0, "1.0", "a pointer is an alia to an object in memori", "The address of a location in memory."
# 705, 1.0, "1.0", "string declar use in an array of charact contain each charact in the array and a special string-termin charact call the null charact versu the type string", "The strings declared using an array of characters have a null element added at the end of the array."
# 707, 1.0, "1.0", "their function signatur", "Based on the function signature. When an overloaded function is called, the compiler will find the function whose signature is closest to the given function call."
# 708, 1.0, "1.0", "it yield the size in byte of the operand which can be an express or the parenthes name of a type", "The size in bytes of its operand."
# 709, 1.0, "1.0", "it take a larg problem and split it into two or more easier or faster solut and make for better readabl", "Divide a problem into smaller subproblems, solve them recursively, and then combine the solutions into a solution for the original problem."
# 710, 1.0, "1.0", "push which insert someth at the top of the stack", "push"
# 711, 1.0, "1.0", "you replac the node with the largest element of it left subtre or replac it with the smallest element of the right subtre", "Find the node, then replace it with the leftmost node from its right subtreeor the rightmost node from its left subtree."
# 712, 1.0, "1.0", "data member local variabl are declar in a function definit s bodi they cannot be use outsid of that function bodi when a function termin the valu of it local variabl are lost", "Data members can be accessed from any member functions inside the class defintion. Local variables can only be accessed inside the member function that defines them."
# 713, 1.0, "1.0", "select sort is an algorithm that select the larg item the array and put it in it place then select the next largest until the array is sort", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 714, 1.0, "1.0", "a function that call upon it self to solv a problem each time it call upon it self it split up a problem into a simplier form until it reach a base case which is the most simplest form of the problem", "A function that calls itself."
# 715, 1.0, "1.0", "it is a type that point to someth els it is the memori address of someth els", "A variable that contains the address in memory of another variable."
# 716, 0.0, "0.0", "2 to the power oflog nn to the power of 2 to the power of 3 loglong nn", "loglog n; 2 to the power oflog n; n to the power of 2; n to the power of 3; n!"
# 717, 0.0, "0.0", "in the main function usual at the top of code they can be declar almost anywher but must be declar befor the code can use or act upon them", "Variables can be declared anywhere in a program. They can be declared inside a functionlocal variablesor outside the functionsglobal variables"
# 718, 1.0, "1.0", "select the smallest number in the list and move it to the front of the list and then advanc to the next number", "Taking one array element at a time, from left to right, it identifies the minimum from the remaining elements and swaps it with the current element."
# 720, 1.0, "1.0", "each node requir an extra node requir more memori and is more difficult to insert and remov individu node", "Extra space required to store the back pointers."
# 721, 1.0, "1.0", "expand the function into the program", "It makes a copy of the function code in every place where a function call is made."
# 722, 1.0, "1.0", "the size of the first dimens can be omit same as a regular array howev for everi dimens outsid the first the size of those dimens must be specifi when pass for exampl a multi-dimension array of 2 4 6 with the name multiarray would be pass as multiarray 4 6 2", "All the dimensions, except the first one."
# 724, 0.0, "0.0", "could travers through the list or array to find the element", "Pop all the elements and store them on another stack until the element is found, then push back all the elements on the original stack."
# 725, 1.0, "1.0", "i would say that a queue is better becaus the first thing you tri to print should be the first one to come out of the printerfifo", "queue"
# 726, 1.0, "1.0", "the do statement first evalu the condit and then execut the line of code in the statement 0 or more time the do while statement execut the line of code and then it evalu the condit", "The block inside a do ... while statement will execute at least once."
# 727, 1.0, "1.0", "variabl can remain privat the code is easili modifi and reusabl as well as easili implement not to mention easier to read and follow along as an observ", "Abstraction and reusability."
# 728, 0.0, "0.0", "you have to keep up with where you are and you have to consid the predecessor and successor connect when insert or delet", "Extra space required to store the back pointers."
# 729, 1.0, "1.0", "it is a sybol or name for a valu or number exampl a use number can stand for ani given number and the programm can refer to that number by use the variabl name", "A location in memory that can store a value."
# 730, 1.0, "1.0", "lognwher n equal the total number of node in the tree", "The height of the tree."
# 732, 1.0, "1.0", "multi-dimension array are store in memori by row", "By rows."
