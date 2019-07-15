# Short Aswer Grading - Capstone Project
Capstone project for Udacity's Machine Learning for Engineers

[Final Report](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/capstone_resport.md)

The purpose of this project is to validate approaches to grading short answer tests using machine learning models. 
The first step was to replicate results of a deep learning LSTM approach with embedding from the Riordan paper (3) reference below.
The results show moderate but limited success and the challenges of deep learning in this domain. Most reaults are in the range of 70-75% accuracy.

This project also implements these approaches in the Sagemaker enviornment and uses Hypertuning to see if the results can be improved.
Finally the results are compared to a simpler XGBoost approach. The results were basically the same as the LSTM approach. 
Because of the inherrent limitation of short answer training data this result wsa not entirely surprising. 
That is to say the results of neither apporach is adequate to automate short answer grading.

Suggested future testing could focus on identifying the kinds of questions and answers that perform better by analyzing the results on a question basis.
Thests should be done with more predecitable short answer questions without typos and spelling issues.
If possible an algorithm to correct spelling could be added to the data preprocessing to make the pretrained embedding more useful.
It is also possible that the results could be used to identify questions that can be autograded while flaging questions or specific answers for human grading if they fall in a range of probabilities that are not as conclusive.

Summary of tests:
  * LSTM with generated embedding
  * LSTM with pretrained Glove embedding
  * XGBoost with simple encoded data
  * XGBoost with additional ngram features.

Additonal Variations for LSTM:
  * Flattening the embedding output
  * Dropout layers
  * Single and Multiple LSTM layers
  
Sagemaker:

* Both LSTM and XGBoost models were trained and tested with Hypertuning in Sagemaker to leverage the scaling and hypertuning capabilities.

Keras/Tensorflow on Sagemaker:

* Local testing for LSTM was done with Keras/Tensorflow which allows for simple model construction. 
* The technical obstacles and limited examples Keras with TensorFlow on Sagemaker proved challenging and the resulting learning will prove useful for future projects.             

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
      

  