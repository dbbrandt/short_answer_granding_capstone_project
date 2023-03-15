# Short Answer Grading - Capstone Project
Capstone project for Udacity's Machine Learning for Engineers

[Final Report](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/capstone_report.md)

The purpose of this project is to validate approaches to grading short answer tests using machine learning models. 
The problem is to determine if a student answer is correct or incorrect based on a model that performs a binary classification or linear regresssion approach. In either case, the output of the model prediction is a probability that an answer is correct or incorrect on a scale of 0 to 1.

The first step in the project is to replicate results of a deep learning LSTM approach (Long Short Term Memory) with embedding from a recent state of the art research paper by Riordan reference below (*3). Each student answer is represented as a vector of numbers, where an integer is uniquely mapped to a word based on a vocabulary of all words in the dataset of answers. Embedding referes to a step where the word vectors are used by the model to create a set output where related words are represented by vectors closer together than unrelated words. This allows the model to handle simiilar words being used to represent the same correct answer. The model can do the embedding at the time its run or pretrained embeddings can be used which can be downloaded from other sources. Pretrained embeddings represent the relationship of words based on a large set of training data that might not be available with a particular problem.
The output of the model is the probability between 0 and 1 that the answer is correct.

The results of state of the art machine learning currently show moderate but limited success automating short answer grading.
Most reaults are in the range of 70-75% accuracy. Something close to 95-100% accuracy is needed. To the extent it cannot be achieved,
the prediction must error toward precision, which means avoiding false positive (grading incorrect answer as correct) or Type 1 errors in favor of Type 2 errors . Answers with a false negative, correct answers marked incorrect,
are easier for a manual grader to review and credit the students test score. Answers graded correct that are incorrect will not likely be reported by students.

This project leverages both local process and the Sagemaker enviornment in order to uses Hypertuning. Hypertuning leverage the power of cloud computer to run a large number of concurrent models at once and find the optimal set of model configurations (tunning parameters) that result in the best predictions. 

This project also compares the state-of-the-art deep learning model, LSTM, to a much simpler but powerful machine learning model, XGBoost. 

The results will show that either approach will not come near to the desired accuracy. That is to say neither apporach is currrently adequate to automate short answer grading. However, deep learning model can be used to partially automate short answer grading and a major goal of this project is to better understand the limitations and strenghts of the deep learning approach.

The results also show that future testing should focus on identifying the kinds of questions and answers that perform better by analyzing the results on a question basis.

Pretrained data was not of much use in this project due to user input errors. Future tests should be done with more predecitable short answer questions without typos and spelling issues. This could be accomplished by adding upfront algorithms to correct spelling and typos. Such preprocessing would make the pretrained embedding more useful. 

Finally, it is possible that the results could be used to identify questions that can be autograded while flaging questions or specific answers for human grading if they fall in a range of probabilities that are not as conclusive.

Summary of tests:
  * LSTM with generated embedding
  * LSTM with pretrained Glove embedding
  * XGBoost with simple encoded data
  * XGBoost with additional ngram features. (Ngram are calculated features that leverage the correct answer compared to the user answer)

Additonal Variations for LSTM:
  * Flattening the embedding output # simplifying the formate of the vectors output by the embedding layer
  * Dropout layers # discarding a fraction of a layers output to reduce overfitting the data
  * Single and Multiple LSTM layers  # using more nodes to handler more complex datasets.
  
Reports:

* See */capstone_proposal.pdf* for the project proposal documentation.
* See */capstone_report.pdf* for the project report.

**1. References**  
  
The project was formulated based on word done for three papers which can be found in the references sub-directory.

1. Suzen, Neslihan & Gorban, Alexander & Levesley, Jeremy & Mirkes, Evgeny. (2019). Automatic Short Answer Grading and Feedback Using Text Mining Methods. Page 1, 19.

2. Galhardi, Lucas & Brancher, Jacques. (2018). Machine Learning Approach for Automatic Short Answer Grading: A Systematic Review: 16th Ibero-American Conference on AI, Trujillo, Peru, November 13-16, 2018, Proceedings. 10.1007/978-3-030-03928-8_31. Page 380

3. Brian Riordan, Andrea Horbach, Aoife Cahill, Torsten Zesch, and Chong Min Lee. 2017. Investigating neural architectures for short answer scoring. In Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications. pages 159–168. Page 159


**2. Source Data Sets**  

The datasets used for testing include two data primary sources. These are located in the /data/source_data directory.

1.	SciEntsBank (SEB) dataset. This data was taken from Dzikovska et al., 2012.  The SciEntsBank (SEB) dataset consists of science assessment questions and I will work the set with 2-way labels (correct/incorrect). *1

2.	Short Answer Grading: University of North Texas short answer grading data set. It consists of ten assignments between four and seven questions each and two exams with ten questions each. These assignments/exams were assigned to an introductory computer science class at the University of North Texas. The student answers were collected via an online learning environment. The data set as a whole contains 80 questions and 2273 student answers. The answers were scored by two human judges, using marks between 0 (completely incorrect) and 5 (perfect answer). Data set creators treated the average grade of the two evaluators as the gold standard to examine the automatic scoring task.*2

**3. Preprocessed Data**

The */data/seb* and */data/sag2* directories contain files used by the models and realted reporting utilities. 

The source data which is stored in a more complex multi-file format have been simplified and stored in csv files.
The two key files are questions.csv and answers.csv. These two files are the primary input for creating train.csv and test.csv files used for modeling.

* questions.csv
  * id - str - unique identifier - can be numeric: ex. seb: 0..n or non-numeric ex. sag: 1.1, 1.10, 1.13)
  * question - str
  * answer - str
* answers.csv
  * Can contain columns for reference the following are used:
  * id - str - question id
  * answer - str - student answer text
  * correct - int or float - 0 incorrect, 1 correct

* Variations of the answer.csv
  * for testing different subsets or additonal features (e.g. ngrams or balanced correct/incorrect)


**4. Data Processing Utilities**

* Found in */source/utils.py* and */source/glove.py*
* Multiple utilities can be found in the /source directory to read the raw source data and generate the preproceesed files.
* Utilies can also be found to generate the train and test data from the preprossed files.
* Finally, functions to build, train and test of models based on a variety of parameters were written.
* */source/train.py* is the Sagemaker training code for the Keras/Tensorflow LSTM model testing
* Ngrams were tested only briefly with XGBoost on SAG data. The code for generating these ng features is in */similarity.py*

**5. Generated Data**

The data is store in the appropriate sub-directory based on it's source (*/data/seb* or */data/sag2*).

* *train.csv*
  * File generated by the load scripts containing encoded answer data for training

* *test.csv*
  * File generated by the load scripts contained encoded answer data for testing
  
* *vocab.csv*
  * Contains the vocabulary used for the encoding. The order of a word in the file is the numeric mapping.  

***Note***: files once generated can be renamed for later use to repete tests. One example is an additional split was done for XGBoost testing where train was split into train and validate and stored as train_xgb.csv annd test_xgb.csv

**6. Model Files**

* Model files after training are stored in the /model directory under the appropriate sub-directory for the data (seb and sag2)

* Two types of persistence are used to handle both TensorFlow and SKLearn/XGBoost model persistence.

**7. Local Model Testing Files**

* Most of the initial testing is done locally using small python script files that setup the parameters and call the data processing utilities which load, process, train and test various models.
* Seb data
    * */seb_tf_train.py* - local training for the LSTM model
    * */seb_xgb_train.py* - local training for the XGBoost model
* Sag data    
    * */sag_tf_train.py* - local training for the LSTM model
    * */sag_xgb_train.py* - local training for the XGBoost model
    * */sag_ng_xgb_trainlpy* - local train test adding additonal ngrams features comparing answer to correct answers
    
**8. Sagemaker Jupyter Testing Files**

* The test and train data can be generated locally and pushed to the github repo for use with the Jupyter files in AWS.
* The Jupyter files were prepared for three tests:
  * Seb LSTM model testing: */Capstone 1 SEB.ipynb*
  * Sag LSTM model testing: */Capstone 2 SAG.ipynb*
  * Sag XGBoost model testing: */Capstone 2 SAG-XGB.ipynb*
  
* Note that the Seb data set is very small so more energy was put into testing with the much larger Sag dataset.    
      

  
