# Machine Learning Engineer Nanodegree
## Capstone Project
Daniel Brandt
July, 10 2019

## I. Definition

### Project Overview

#### Background ####
Automation of grading multiple choice exams is trivial. For many reasons, however, short answer tests provide a better tool for helping students assess their knowledge. So far, this area remains a challenge and machine learning has not yet provided usable solutions.
Several papers have discussed both algorithmic natural language solutions as well as deep learning approches. None of the approaches so far have provided a general pupose solution to the many challenges faced. 
These challenges are in some ways very similar to some of the most important AI projects of our time: Intelligent Assistants like Apple Siri, Amazon Echo and the Google Assistant. 
Given a very broad set of possible subjects, any given question can be phrased in innumerable ways. How do you accurately interpret what the essential question is?

For short answer tests, a given question can be answered correctly in innumerable ways as well. Like with voice assistants, the quality of the response in terms of grammer, spelling, typos and word choice are extreamly variable.
What makes the short answer grading possibly more challenging than large scale intelligent assistants is that each individual test might have very limited responses and may not be in use for that long. 

Because short answer tests are very tedious to grade, the success in automating related areas such as identifying plagiarism and essay grading has led to renewed efforts to attach this area. The problem is challenging because of the need to focus on identifying correctness in relatively short answers. Longer answers and essays require a broader set of criteria like grammar, ideas, structure and therefore provide more features for machine learning to work with.

If we had enough short answer test results graded by a human for a particular test, it would be relatively easy to model and grade a short answer test. Unfortunately, exhaustive student answer datasets rarely exist in the real world for a specific short answer test. The reality is that tests generally have to be refreshed regularly to reflect constantly changing content and, in some cases, to prevent unwanted distribution and cheating. 

**1.**  Some of the Challenges of Grading Short Answer include:
  *	Teachers usually find the task of assessing respondents’ answers very time-consuming.
  *	Students may have to wait for a long time to receive feedback on their responses 
  * When they finally get it, the grade can be different from another classmate’s, who has given a very similar answer. *2

**2.** The challenges of grading Short-Answer compared to essays are: 
  * Response length. Responses in SAS tasks are typically shorter. For example, while the ASAP-AES data contains essays that average between about 100 and 600 to-kens (Shermis, 2014), short answer scoring datasets may have average answer lengths of just several words (Basu et al., 2013) to al-most 60 words (Shermis, 2015).
  * Rubrics focus on content only in SAS vs. broader writing quality in AES.
  * Purpose and genre. AES tasks cover persuasive, narrative, and source-dependent reading comprehension and English Language Arts (ELA), while SAS tasks tend to be from science, math, and ELA reading comprehension. *3

**3.** Defining correct:
* From my analysis, I believe one of the most difficult challenges is defining what correct actually means. There is no absolute measure of correctness for this kind of test, rather, we can only look at how human graders make the judgement. Human graders tend to look for key words and pattern matching rather than the specific order of words when grading large numbers of questions. 
  * The number of correctly used words has more influence on marks than semantics or order of words. 
  * If a large number of responses are being graded, it is not unreasonable that a human would move towards pattern recognition via key words rather than “reading for meaning”. 
  * Identifying words gives an idea about grades and students misunderstanding to teachers. Such an approach allows time saving for scoring, and to provide rapid feedback to students by checking the words used from model vocabulary. *1

The ideal case would be a solution that can use a subset of answers by the first batch of students that are manually graded to then automatically grade subsequent tests with the same questions.
The goal of this project is to test that approach on some datasets with baseline results in academia to gain a better understanding of the problem and the build a baseline of code to iterate on better solutions.

A complete solution is not the only valuable outcome. Machine learning can still add a lot of value today. While complete grading may not be possible, automation of some of the answers allowing the grader to focus on a smaller supset could also be a win. 
This exercise may also help design short answer tests to make them easer to automatically grade. For example, a recent paper recommended that it be used to supplement the quality, provide automation in some areas of grading and help target areas where more human involvement is needed. *1

This project originated from my interest in both memomry and how memory impact learning. We employ two types of memory while learning: recognition and recall. 
A multiple choice test involves a larger percentage of recognition as the answer is provided and the correct one will be recognized. Short answer on the other hand requires a more intense form of memory where with no prompt an answer must be recalled.
This second form of question is often easer to write but much harder to grade. It's easier to write because random but possibly related answers must be generated. 

#### Overview ####

The goal of this project is to build the tools and models necessary to demonstrate the capabilities of various apporaches of machine learning to grade short answers. 
The first step is to identify literature that show the current state of the art in solving this problem. This includes identifying baseline models and datasets from which we can validate the code developed in this project are valid and reproduce the results from the literature.

In addition to implementing the models and utilities to load and process the test data, we also want to deploy the solution in Sagemaker so we can leverage the power of cloud computing and Hypertuning to see if we can imporove on the results of the literature.
The primary reference models used were based on an deep learning approch used in the Reordan paper (*1) cited above. 

Reordan was basically an LSTM model with an embedding layer. The embedding layer accepts encoded english sentences and then converts them to word vectors that improve the representation of the words for machine learning solutions.
The use of the LSTM model leverages the fact that sentences have sequential wording and order of words helps interpret the meaning. 

After completing all the testing with a deep learning approach, a more simple XGBoost approach is tested and tuned with the same data to see if the LSTM approach was significantly better.

To cover a variety of short answer data challenges, two different dataset were chosen from different research papers:

1.	SciEntsBank (SEB) dataset. The dataset consists of science assessment questions with 2-way labels (correct/incorrect). *3 (Reordan) 

2.	Short Answer Grading (SAG): These assignments/exams were assigned to an introductory computer science class at the University of North Texas and werecollected via an online learning environment.  
*1 (Suzen)

#### Reference ####

*  **1** Suzen, Neslihan & Gorban, Alexander & Levesley, Jeremy & Mirkes, Evgeny. (2019). Automatic Short Answer Grading and Feedback Using Text Mining Methods. Page 1, 19.

*  **2** Galhardi, Lucas & Brancher, Jacques. (2018). Machine Learning Approach for Automatic Short Answer Grading: A Systematic Review: 16th Ibero-American Conference on AI, Trujillo, Peru, November 13-16, 2018, Proceedings. 10.1007/978-3-030-03928-8_31. Page 380

*  **3** Brian Riordan, Andrea Horbach, Aoife Cahill, Torsten Zesch, and Chong Min Lee. 2017. Investigating neural architectures for short answer scoring. In Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications. pages 159–168. Page 159

___

### Problem Statement

#### Goal ####
Create a machine learning model and the supporting code to perform grading of an arbitrary short answer test given a limited set of actual test results. 

For this project I will focus on reproducing some results from the literature and implementing an approach with Sagemaker and a custom Sklearn deep learnign LSTM model.
For comparison a basic XGBoost model will also be tested with the same data. 

The problem will be broken into two pieces. 

First, I will try to replicate the results from the Riordan (*3) paper referenced above using Sagemaker with hypertuning and a basic LSTM model. 
The paper described several additional layers and techniques in addition to basic LSTM.  They identify improvements to a basic model such as pretrained turned embedding and an attention layer may. 

The desired outcome is to achive the most accurate prediction of correct or incorrect for the student answers. 
The current state of the art from the Reordan paper achieves results with around 75% accurancy for the data sets used in the project.
Ideally this bench mark can be matched or exceeded.

#### Approach ####

Once the two types of models are built, tuned and tested with the relatively small SEB data set of 4 questions, 
the same models and tuning will be applied to an entirely different and larger data set, Sag2, which consists of 88 computer science questions and about 2500 answers. 

The approach is to recreate the basic model variation from the Riordan paper which leverages LSTM to see the results for the SEB data. 
LSTM make sense here because the order and intent of the words is important and language based datasets are often well handled by adding memory to the model. 

The paper also suggestes a variety of adjustments to the model such as pretrained embedding that will be experiment with to see if they improve the results.

### Metrics
This is a basic binary classification problem so the metrics are pretty straight forward. We are looking for accuracy. In addition recall and precision are relevant as they help indeify the common case of predicting mostly correct or incorrect. 
We saw this often when models failed to converge on a solution. Also, for short answer grading, there is some preference for precision over recall because it's better to err on the side of incorect which can then be manually graded to imporve a score. 
This is always preferrable to reducing a students grade.

The key metrics are:

* True Positives - np.logical_and(test_labels, test_preds).sum()
* False Positives - np.logical_and(1 - test_labels, test_preds).sum()
* True Negatives - np.logical_and(1 - test_labels, 1 - test_preds).sum()
* False Negatives - np.logical_and(test_labels, 1 - test_preds).sum()
* Recal -  tp / (tp + fn)
* Precision - tp / (tp + fp)
* Accuracy - (tp + tn) / (tp + fp + tn + fn)

___

## II. Analysis

### Data Exploration

The datasets used for testing include two primary data sources. These are located in the /data/source_data directory.

---

**1**.	**SciEntsBank** (SEB) dataset. This data was taken from Dzikovska et al., 2012.  The SciEntsBank (SEB) dataset consists of science assessment questions and I will work the set with 2-way labels (correct/incorrect). *3 (Reordan)
  * The data is stored in XML format, one file for each of four questions.
  * There are 4 files representing 4 questions. Each question has approximately 30 to 40 answers.
  * The ratio of correct to total is 37.5%. This indicates that the majority of the students did very poorly. Examining the data shows some very confused respones. 
  * Each file includes the questions text, correct answer text and a list of graded answers (correct/incorrect)
  * To make processing easier:
    * **question data** (question_id (0..3), question text, reference answer) were combined into a single file, **questions.csv**
    * **answer data** (question_id, answer text, score - 0/1)  were combined into a single file, **answers.csv**
  
  * Example Question Data
    * **Question:** (id=0): Carrie wanted to find out which was harder, a penny or a nickel, so she did a scratch test. How would this tell her which is harder?
        * **Reference Response**: The harder coin will scratch the other.
        * **Correct**: The one that is harder will scratch the less harder one.
        * **incorrect**: She could tell which is harder by getting a rock and seeing if the penny or nickel would scratch it. Whichever one does is harder.
        * **incorrect**: Rub them against a crystal.

    * **Question:** (id=2) A solution is a type of mixture. What makes it different from other mixtures?
        * **Reference Response**: A solution is a mixture formed when a solid dissolves in a liquid.
        * **Correct**: It dissolves the solid into a liquid that is see through
        * **incorrect**: A solution is a different type of mixture. Then they are a solution is a mixtures that dissolves.
        * **incorrect**: When the mixture is mixed.    

---

**2**. **Short Answer Grading**: University of North Texas short answer grading data set. 
These assignments/exams were assigned to an introductory computer science class. 
The student answers were collected via an online learning environment. 
The answers were scored by two human judges, using marks between 0 (completely incorrect) and 5 (perfect answer). 
Data set creators treated the average grade of the two evaluators as the gold standard to examine the automatic scoring task. *1 (Suzen)

  * The data set as a whole cont ains 80 questions and 2242 student answers, or about 30 per question.
  * The ratio of correct to total answers is about 71%. This makes sense as you would expect an average passing grade.
  * While this is a sligh imbalance, testing results did not show a significant advantage to balancing the test data 50/50 correct/incorrect.
  * Answer lengths varied between 1 and 950 words with the bulk in the 25 to 100 word range. 
  * The data set is stored in a very complex to process format of multiple text files and sub-directories. 
  * Multiple versions of the same data is available in aggregated and file per question formats.
  * Score were not aggregated into a single file and had to be aggregated using a script.
  * Scores were in a range of 0 to 5 and were converted based on a qualitative decision as correct if >= 4. Another option would have been to chose 3, the midpoint as the cuttoff or any other value > 0.
  The criteria for the decision would be ideally done by the question creator. Here the goal was to have a sufficient balance of correct and incorrect and try and filter out the more confusing answers which presumably have a lower score.
  * The data has a number of text representations for punctuation that were removed for this exercise. ex. -LRB-, -RRB- and <STOP>
  * The files used for this project were:
    * **Questions** - *ShortAnswerGrading_v2.0/data/sent/questions* in the format of questions_id  and question text seperated by a space
    * **Reference Answers** - *ShortAnswerGrading_v2.0/data/sent/answers* in the format question_id and answer text (spaced)
    * **Student Answers** -  *ShortAnswerGrading_v2.0/data/sent/all* in the format question_id and answer text (spaced)
    * **Files** - *ShortAnswerGrading_v2.0/data/docs/files*. A single list of question_ids/filenames which identify the directories for scores by question_id.
    * **Scores** - Using the Files list, all scores were concatenated in order from the files located in *ShortAnswerGrading_v2.0/data/scores/<question_id>/ave where question_id was taken from the files list.
    
  * To make subsequent processing easier
    * **question data** (question_id, question, reference answer) were combined into a single csv file, **questions.csv**
    * **answer_data** (question_id,answer, score) were combined into a single csv file, **answers.csv**  
  * The simplified questions.csv and answers.csv are stored in */data/seb* and */data/sag2* respectively. 

    **Example Question Data**
    * **Question:** (id=1.1) What is the role of a prototype program in problem solving?
        * **Reference Response**: To simulate the behaviour of portions of the desired software product.
        * **Correct**: you can break the whole program into prototype programs to simulate parts of the final program.
        * **incorrect**: To lay out the basics and give you a starting point in the actual problem solving.

    * **Question:** (id=5.1) In one sentence, what is the main idea implemented by insertion sort?
        * **Reference Response**: Taking one array element at a time, from left to right, it inserts it in the right position among the already sorted elements on its left.
        * **Correct**: insertion sort is were after k iterations the first k items in the array are sorted it take the k plus 1 item and inserts it into the correct position in the already sorted k elements.
        * **incorrect**: Take a number and choose a pivot point and insert the number in the correct position from the pivot point.    

    * **Question:** (id=7.2) What is the main advantage of linked lists over arrays
        * **Reference Response**: The linked lists can be of variable length.
        * **Correct**: Array size is fixed, but Linked is not fixed.
        * **incorrect**: Linked lists have constant time insertion and deletion
        * **incorrect**: There is no limit as to how many you create where an array can only hold a given amount of information.
            * Note: This second incorrect answer is provided here because I would grade it as correct. This highlights the human error potential in grading which adds an additional random factor to the data.

    * **Question:** (id=11.1) What are the elements typically included in a class definition?
        * **Reference Response**: Function members and data members.
        * **Correct**: data members and function definitions.
        * **incorrect**: Class name,, semicoln at the end of the defination, private and bublic followed by :   

    * **Question:** (id=12.10) How many steps does it take to search a node in a binary search tree?
        * **Reference Response**: The height of the tree.
        * **Correct**: to find a node in a binary search tree takes at most the same number of steps as there are levels of the tree.
        * **incorrect**: it depends on the install search tree then from there for whatever the case is the it repeats it back along the case of the primary node.    

---

In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization

The data used in this project has been well vetted as a good use case for short answer grading in the papers cited at the top of this report.

The feature or features used are essentiall the encoded representation of strings that were provided as answers. This enconding is achieved by creating a word dictionary and assigning the index from the dictionary to the word.
The feature is then a list of integers representing the words.  The dictionary size for the small Seb dataset is about 265 words. While the dictionary size for the large Sag2 dataset is about 2650 words.

What is most import to visualize is the grading of the answers. If for example 95% of the answers were correct or incorrect, we whould need to consider this a unbalanced dataset and adjust for that. 
The visualization show that the data, while not exactly balanced, has a signifiant result set for both correct and incorrect. In both cases is a 2/3 to 1/3 ratio, For the seb dataset, that ratio is in favor of incorrect while for Sag2 in favor of corret.

The Sag data include one addition complexity with respect to grading. The raw answers were scored on a range of 0-5 and were an average of two graders. The restult are scores from 0 to 5 in increments of .25.
The final decision on how to convert that to a binary representaion of correct and incorrect was somewhat sugjective. One of the major considerations was to balance the data. The lower the correct cutoff chosen was, the smaller the fraction of incorect answers would be. 
There was, therefore, an argument for making this as high as possible while stil making sense from a usecase point of view. I ended up chosing a score of 4 as the cutoff. This translates to an 80% grade which seems reasonable for passing. 
It is quite possible that additional analysis and testing with other cutoff, particulary with a wider range of test data sets, would require some more complex modeling and testing to determine decision criteria. This is out of scope for this project. 

I have also included a distribution of answer lengths. This information is likely to play into the effectiveness of the model fitting. It has been documented in the research papers that shorter answers are more difficult to fit. 
From the historgrams we can see that a majority of the answers for both datasets fall in the range of 50 to 100 words. This confirms the relatively short nature of the answers as compared to essay or even short essay type response for which other modeling technical have proven successful (see *3 Reordan )
 
One final area that bears noting is the difficulty of the questions. This can be shown in a graph of the percentage correct of the answers by question from the result set. The chart shows that while the average is around 80% it varies widely between 20 and 100%. 
This is likely to have an impact on the predictability of the data. This mean each questions data is significantly different in composition. 
**1**.	**SciEntsBank**

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Seb-Correct-Incorrect-Histogram.png?raw=true)

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Seb-Answer-Length-Histogram.png?raw=true)

**2**. **Short Answer Grading**

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag2-Correct-Incorrect-Histogram.png?raw=true)  

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag2-Answer-Length-Histogram.png?raw=true)

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag2-Answer-Raw-Score-Histogram.png?raw=true)

Note: As mentioned, with this disrtribution of answers and the cutoff for correct wsa set at 4.0 resulting in about 1770 correct and 670 incorrect. 

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag-Answer-Correct-Percent-Distribution.png?raw=true)

Note: The distribution of question difficulty is pretty even and wide, ranging from 20% to 100% answering correctly.


In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques

*Data Pre-Processing*: The first non-trivial task was to preprocess the raw datasets some of which are XML based and generate encoded train and test input datafiles.
On of the goals of the encondining was to build in the ability to decode the results once broken up and randomized into train and test sets. 
This decoding helps to better understand and visualize the challenges with some of the answers as compared to others. For that, the question_id
is added to the embedding vector for the answer. While not explored here, include a feature for which question an answer relates too could help identify similar answers. In practice, the answers are so varied, 
it is unlikely this one integer in a long embedding vector will have any impact. It value is in analyzing the results by making it easy to compare correct and incorrectly predicted ansewrs to the correct answer to see
if any patterns emerge to what is predictable. Most of the data processing code is found in */source/utils.py*. A lot of methods are in this one file and it would make sense to break them out into fiels specific to the two data sources and another for modeling. That was not yet done at the time of this writing.

To simplify working in the AWS Sagemaker enviroment, the generation of test.csv and train.csv was done during local testing and the committed to Github for use
in the Sagemaker Jupyter notebooks. It would be simple add calls to the methods in /source/utils.py to make the Jupyter notebooks fully self contained. It was not done at the time of this writing.

*Modeling*: The next step is to building the model and training code for the LSTM and XGBoost models.
A version for local testing was done first and then translated to work on SageMaker with Jupyter for Hypertuning. 

#### LSTM - Deep Learning ####
The basic framework for the LSTM deep learning model comes from the Reordan (*3) paper. They tested a number of variations some of which were not effective. For eaxample they added a convolution layer after an embedding layer 
but found it of limited value so it was not tested here.

The basic model use here consists of:
* An embedding layer that process the encoded sentences and produces word vectors for each word in the vocabulary.
    * After testing the generated embeddings some effort was also put into use pretrained embeddings from Glove data file with 50 dimensions (data/glove/glove.6B.50d.txt).
    * This required deriving a dictionary from the Glove file by selecting the words from answer dictorionary  and using the glove id's to lookup the embedding_matrix and creating a custom_embedding_matrix for our dictionary.
    * The pretrained custom embedding matrix was used to preload the embedding layer. The code for this can be found in */source/glove.py*.
    * One challenge to note is that mispellning and words or garbage input was not found in the glove database and 0's were used insetead. This represented about 18% of the words. Not insignificant portion.
    * Not in the scope of this project was to correct misspelled or concatenated words to reduce the glove databases misses. It is outside of the scope of this project to test these complex cleanup approaches.  
* An LSTM layer with one or more sub-layers to model more complexity.
* Dropout layers to help tuning and beter managing overfitting. This proved quite valuable when dealing with problems of models converging at all. This would reqire overfitting and then dropout increases to make the test predictions more accurate.
* Activation layer and Objective. Linear activation rather than sofmax/binary_crossentropy as might be expected becuase it was describes as the more successful approach in Reordan. This was validated with some initial tests. It also has the benefit of giving more visability into the probabilities.
I should be noted that for XGBoost, binary:hinge truned out to be the best approach. This suggests that using linear may be a symptom of the overall poor results obtained for this problem set using deep learning methods.

The figure below is visualization of the model approach. 

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Basic%20LSTM%20Model%20Diagram.png?raw=true)


*Tuning Parameters and Variations*: In addition to the basic model above, some variations and adjustments to the model are tested. These variations were documented in the Reordan paper as having potentially meaningful impacts on success. 
The key ones tested in this project are:
* Pretrained v.s. Generated embeddings. 
* Flatening and Shaping (Yes or No). These layers can reduce overfitting or improve fitting in general. The complex embedding layer which is 2D for each entry can be flattened. This suggestion was found while researching embedding layers and a case was made it could help.
* LSTM layers and layer sizes. One or two layers of varying sizes.
* Dropout layers: Two drop out layers after embedding and LSTM with variable dropout percentages. 

**Sagemaker and Keras LSTM**

The LSTM models required custom coding to work in SageMaker. For ease of local testing I chose to use as Keras Sequential model which uses Tensorflow as the engine.
While it was quite straight forward to do this locally, quite a bit of research and trail and error were required to get the custom model working in Sagemaker. Additional work was also needed
to capture the necessary metrics from the training jobs for use by the Hypertuning process. This work while ultimately no a lot of code was non-trival. The resuls provide a useful framework for future
testing and an easy path to move quickly developed Keras models from local to Sagemaker implementations. 

#### XGBoost ####

As a comparison and sanity check, XGBoost was run on the same train.csv and test.csv files used with LSTM. The usual tunning parementers were used to best optimize the model for the short answer data.
The initial values were selected from similar examples used in sentiment analysis in the course. After some manual tuning locally the ranges for hypertuning were selected and then adjusted after the first runs.

### Benchmark

The benchmark for this project was well documented in the Reordan paper for a variety of datasets they used. The table provided below include two datasets they tested with both 2 and 5 way grading. They also have their benchmark and the resutls of all the variations of modeling they used.
The net results all fell in the range of 70-75% acurancy which is the target accuracy this project used as a bench mark. The highlighed section in yellow is the first of two datasets tested in this project referred to here as SEB. 

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Benchmark-SEB-2Way-Riordan.png?raw=true)


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

The preprocessing of data was done in such a way so that both data source (SEB and SAG2) generated identically formated question and answer datafiles.
The goal of the preprocessing was to generate the following file formats which represent the minimum fields required. Additional fields are allowed for analysis purposes but will not be used to generate the train and test files.

* Questions (questions.csv)
    * question_id - unique indentifier for the question
    * question - text of the question
    * answer - text of the reference answer
    
* Answers (answers.csv)
    * question_id - unique identifier for the question
    * answer - text of the user response
    * correct - 0 or 1

    Notes:
    * For SEB which was used during the first testing, question text and reference answer were also store to help with debugging. 
    * For SAG the score was also stored (0-5) for later analysis purposes

#### SEB Preprocessing ####

The preprocessing is all found in one method in */source/utils* called *load_seb_data(). 

**File Processing**

This method read the original raw source XML and writes the standard
question.csv and answer.csv for use in analysis. 

As described in the **Data Exploration** section, SEB data for the four questions is stored in four XML files which each contain all the data for the question. 
This includes question text, reference answer, student answers and scores. Because XML is fairly easy to process, the code was written to load the data direcly from the XML files each 
time the model was tested. The files were also quite small so this provided little overhead. The question.csv and answer.csv files are generated on each run

Several additional helper funtions that leverage *xml.dom.minidom* were written to extract the desired fields from the SEB question files (*read_xml_qanda()* and *get_xml_elements()* ).

**Answer Encoding**

The next step is to generate a vocabulary and map the text words to list of numbers. This step must also determine the longest sentence to support padding later on so that all answer vectors are the same length.
The vocabulary is also used for decoding the encoded answer vectors after the results of prediction to help with evaluation of the model.

The enconding is done by the fuction, *generate_data()*,  in /source/utils. This calls an number of other methods that generate the vocuabulary and map the words to that vocabulary. 
The vocabulary is generated by concatenating all the answers into a single text string, cleansing the string and creating a word dictionary with unique integer ids.

The encoded answer vectors are padded based on the longest answer found in the test data.

The data is then randomized by adding the predictor column (correct answer) to the answer vectors and using the DataFrame reindexing feature with np.random.permutations.

The data is then split back into answsers and labels and passed back to load_seb_data().

**Train and Test Data**
The final step is to use sklearn.model_selection.train_test_split using a 30% split to generate the  X_train, y_train, X_test and y_test data.
These along with the vocabulary and the length of the answer vector needed by the Embedding layer are returned for use in the modeling phase.
This data is also stored in train.csv and test.csv with the label prepended in the first column for use in Sagemaker models.


#### SAG Preprocessing ####

Because of the complexity of the SAG raw data as documented in the **Data Exploration** section above, the raw data was process once with conversion to csv and a set of
python commands to generate the question.csv and answer.csv. Once completed the creation of the train and test data is done using the same approach as with the SEB data above.

*Manual Steps*

1. The original data files are broken into separate files for
    * Questions: data/ShortAnswerGrading_v2.0/data/sent/questions
    * Correct Answers: data/ShortAnswerGrading_v2.0/data/sent/answers
    * Scores: data/ShortAnswerGrading_v2.0/data/scores/1.1/ave.  (one file per question)
    * Answers: data/ShortAnswerGrading_v2.0/data/sent/1.1 (one file per question)

2. Combine the questions and answers into a single file: /data/sag2/questions.csv
    * Manually converted questions and answer files which were spaced seperated into csv format in an editor
    * Used Pandas to merge the two files into one.
    * Note that the id field is in a format that will be misinterpreted as a number resulting in the loss of information for ids like 11.10 and 12.10.
    To handle this we insure that pandas treats it as a string.
        
       
    questions = pd.read_csv('data/sag2/questions.csv', dtype={'id': str})
    questions.set_index(‘id’)
    answers = pd.read_csv('data/sag2/answers.csv', dtype={'id': str})
    answers.set_index(‘id’)
    qanda = pd.merge(questions, answers)
    Deleted the working questions and answers separate files and replaced with combined file
    qanda.to_csv('data/sag2/questions.csv', index=False)


3. Create a single data_file for answers. This includes questions_id, answer responses (sentences) and scores.
    * Wrote a script to read and concatenate the answers and scores using the names of the files (data/ShortAnswerGrading_v2.0/data/docs/files)
    * The complete list of all answers is store in a single file *data/sag2/sent*
    * All the scored must be read question and concatenated in the order of the answers.
    * They are then appended as a series to the answer set.


    files = pd.read_csv('data/ShortAnswerGrading_v2.0/data/docs/files’, dtype={‘id’:str})
    scores_df = pd.DataFrame(columns = [‘id’, ‘score’])
    for id in files['id'].values:
         scores[’score’] = pd.read_csv(f"data/ShortAnswerGrading_v2.0/data/scores/{id}/ave”)[0]
         scores[‘id’] = id
         scores_df.append(scores, ignore_index=True)

    answers = pd.read_csv('data/sag2/sent', header=None, dtype={'id':str})
    answers['score'] = scores_df['score']
    answers[‘correct] = answers[’score’] >= 4
    answers.to_csv('data/sag2/answers.csv', index=False)
 
   

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation

The implementation is based around driver python scripts that configure the hyper parameters and call methods in the */source/utils.py* script to load data and execute the model training and evaluation.

All local LSTM models are based on SkLearn with Keras on top of Tensorflow. The Keras Sequential model is used to build the model layers.
The local XBGoost uses XGBCLassifier which conforms to the SKLearn api.

All Sagemaker models are based on SageMaker classes. The Keras/Tensorflow approach used Locally for LSTM is not diretly supported on SageMaker so we use a SageMaker Tensorflow class and call a custom train.py file (located in /source/train.py) which builds and trains the Keras LSTM model. This train.py also generates output that can be parsed by SageMaker to obtain the metrics needed for Hypertuning.    

#### Local Testing ####
Most of the initial testing was done locally on a laptop to debug and get initial tuning parameters. The goal was also to eliminate low performing configurations before using SageMaker to try Hypertuning. Each driver script file for local processing is setup for a dataset and and model type combination.   

As much as possible all of the testing options were coded as options that could be passed in to the utilities to quickly test alternatives.
The selection of dataset and model type are the only exception. Since the features and parameters that work for one data set and model may not work or be applicable to another, each are tested as a seperate work path with a seperate driver script.

The scripts for SagemMaker testing are basically Jupyter Notebooks. All LSTM which 

**Local Tests**

1. SEB Data with LSTM (*seb_tf_train.py*)
2. SEB Data with XGBoost (*seb_xgb_train.py*)
3. SAG Data with LSTM (*sag_tf_train.py*)
4. SAG Data with XGBoost (*sag_xgb_train.py*)
5. SAG Data with XGBoost and added ngram features (*seb_ng_xgb_train.py*)

*1 A theory was tested that adding ngrams, a comparison of student answer to reference answer, to the feature set might help the model. This is similar to sentiment analysis.

The **local mode** code as well as the Sagemaker code passes in the tuning parameters as a dictionary into the common shared code. In the case of local processing, the common shared method from */source/utils*  named *train_and_test()* for LSTM modeling. 
This method build the model, fits the data and then uses the test data with the model's predictor to evaluate the model accuracy. 

A similar apporach is used for XGBoost with the slight variation because the persistence of models is different between Keras and XGBoost. To handle this some of the processing is moved to the driver scripts and the lower level methods used by *train_and_test)_* are caled directly. In this case *build_mode()* which returns the model is called directly and model.fit() is called in the driver script and the resulting model is then saved using a *save_xgb_model()* method. Finally the evaluate() and display_results() are called in the driver script. 

#### Local LSTM ####
Tunning Parameters for **local mode** testing:

The tunning parameters passed into that method are:


    model_params = {'max_answer_len': max_answer_len,   # use by the embedding layer
                    'vocab_size': vocab_size,           # used by the embedding layer
                    'epochs': 20,          
                    'pretrained': pretrained,           # used by build model to load petrained embeddings if true
                    'embedding_dim': 50,                # Size of embedding dimension
                    'flatten': True,                    # The embedding dimension is flattened and shaped if true
                    'lstm_dim_1': 100,                  # Size of the first lstm dimension (> 0)
                    'lstm_dim_2': 20,                   # Size of the second lstm dimension (>=0, 0 means none)
                    'dropout': 0.3}                     # The dropout layer percent used (>0).


These can be found in the codebase **/source/train.py**  and the Jupyter files,**Capstone 1 SEB.ipynb**,  **Capstone 2 SAG.ipynb** for the two data sets.

#### SageMaker Testing ####

The **SagemMaker mode** code is based on Jupyter Notebooks that setup SageMaker models. The XGBoost model works as a standard SageMaker build in model. For LSTM, a TensorFlow model is used which calls custom code to build and train the model. The code persists the model as expected by SageMaker and also calculates the needed metrics for parsing by SageMakers Hypertuning jobs.

**SageMaker Tests**

1. SEB Data with LSTM/Tensorflow - (*Capstone 1 SEB.ipnyb*)
2. SAG Data with LSTM/Tensorflow - (*Capstone 2 SAG.ipnyb*)
3. SAG Data with XGBoost - (*Capstone 2 SAG-XGB.ipnyb*)

Each of these Notebooks reads the data already prepared by the local processing and stored in the git repo under /data/seb and /data/sag in train.csv and test.csv.
It would be easy to call the utilities to build these files from the Notebooks but since they were already generated locally a simple push and pull from the git repo was used. Further work could be done to make the Jupyter Notesbooks fully self contained but this was not in scope for this project.

The data is loaded into notebook and then uploaded to S3 buckets for use by the SageMaker models. 
The model was first tested with just an estimator to verify it was working and then a Hypertuning step was done to trying at optimize the parameters. Finally an endpoint is created and the test data is then evaluated using an *evalutate()* function which calls the predictor and generates the desired metrics.

The tunning Parameters passed into the Tensorflow model are very similar to those use for the local mode with the exception that pretrained was not tested because it did not prove helpful and infact was the opposite in local testing:

*Ngrams* 

One of the experiments mentioned was to try adding Ngrams to the features. This was tested with XGBoost rather than LSTM becuase of the alrady complex Embedding layer logic. The ngrams were calclated using the *calculate_containment()* method found in the */similarity.py*. This code was most lifted from the ealier plagiarism project.  That project also did an LCS (longest common sequence) match. Due to the short lenght of many answers, this didn't seem appropriate for short answer daatasets.
Average containment for multiple values of n were calculated and not surprisingly only n1 and n2 made sence with the short answers. A cross correlation betwen n1, n2 and the question correct label were completed to see how well each correlated. The selection of ng1 and ng2 was made and a test and train data set was manually created by appending these values to the existing generated train.csv and test.csv.
While the results were not that useful with the models selected here, they are show in the results section and may be an avenue of exploration for future approaches.   

**Custom Parameters**

Based on local testing slighly different tuning parameters were used based on learning from those tests realted to the difference in data.
See Local Testing for definition of paramters which are essentially the same.

**SEB**

       hyperparameters = {
        'epochs': 200,
        'embedding_size': 30,
        'flatten': 0,   
        'lstm_dim_1': 100,
        'lstm_dim_2': 0,
        'dropout': 0.2
       
**SAG**

       hyperparameters = {
        'epochs': 200,
        'embedding_size': 50,
        'flatten': 1,   
        'lstm_dim_1': 100,
        'lstm_dim_2': 20,
        'dropout': 0.2        


HyperTuning Parameters for Tensorflow:
Note: We also show how the metrics were parsed from the train.py output. Validation_accuracy is the optimization metrics selected.


**SEB**

    tf_hyperparameter_tuner = HyperparameterTuner(estimator = estimator, 
           objective_metric_name = 'Validation_accuracy', # The metric used to compare trained models.
           objective_type = 'Maximize', # Whether we wish to minimize or maximize the metric.
           metric_definitions = [{'Name': 'Validation_loss', 
                                  'Regex': 'Validation_loss:(.*?);'},
                                 {'Name': 'Validation_accuracy', 
                                  'Regex': 'Validation_accuracy:(.*?);'}
                                ],
           max_jobs = 18, # The total number of models to train
           max_parallel_jobs = 6, # The number of models to train in parallel
           hyperparameter_ranges = {
                'dropout': ContinuousParameter(0.1, 0.4),
                'embedding_size': IntegerParameter(20, 100),
                'lstm_dim_1': IntegerParameter(50, 150)
           })        

**SAG**

    tf_hyperparameter_tuner = HyperparameterTuner(estimator = estimator, 
           objective_metric_name = 'Validation_accuracy', # The metric used to compare trained models.
           objective_type = 'Maximize', # Whether we wish to minimize or maximize the metric.
           metric_definitions = [{'Name': 'Validation_loss', 
                                  'Regex': 'Validation_loss:(.*?);'},
                                 {'Name': 'Validation_accuracy', 
                                  'Regex': 'Validation_accuracy:(.*?);'}
                                ],
           max_jobs = 18, # The total number of models to train
           max_parallel_jobs = 6, # The number of models to train in parallel
           hyperparameter_ranges = {
                'dropout': ContinuousParameter(0.1, 0.3),
                'embedding_size': IntegerParameter(50, 200),
                'lstm_dim_1': IntegerParameter(50, 200),
                'lstm_dim_2': IntegerParameter(10, 50)
           })


#### XGBoost Parameters ####

XBGoost was only used to test SAG data as a reference to see how a powerful machine learning algorithm compared to deep learning approaches.
The basic starting parameters were dervied from local testing and then Hypertuning ranges were adjusted based on tunning runs.

**SAG**

Estimator Parameters:

        max_depth=5
        eta=0.2 
        gamma=4
        min_child_weight=6
        subsample=0.8
        objective='binary:logistic'
        early_stopping_rounds=50
        num_round=4000
 
Hypertuning Parameters:

        'max_depth': IntegerParameter(3, 12)
        'eta'      : ContinuousParameter(0.05, 0.5)
        'min_child_weight': IntegerParameter(2, 8)
        'subsample': ContinuousParameter(0.5, 0.9)
        'gamma': ContinuousParameter(0, 10)

In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement

The refinement process consisted of three major phases:
(See the **Model Evaluation and Validation** section for additonal refinement details)

*LSTM First*
Since the baseline is using the LSTM model, all initial tests were with variations of that model.

1. Start testing locally to get some idea of the parameters and ranges of values that have a positive impact.

2. Get a baseline model to converge on training accuracy of 100%, essentially through overfitting as required.

3. Adjust the basic parameters individually to see the directional impact they have on accuracy. These include epochs, dropout, LSTM size, and Embedding size.

4. Try some of the model changes proposed to see if they improve or match the basic model adjustments. If they are competitive consider them for further testing. These include, flattening, pretrained embeddings, multiple LSTM layers.

5. Take the most promissing values and model variations and build a SageMaker impelemtation for validation, testing and then Hypertuning.

6. Validate that the local results can be replicated.

7. See if better results can be obtained with Hypertuning that were done with local testing.

**XGBoost**

1. Using the same data use for LSTM, test an XGBoost model locally.

2. Do some basic tuning tests and compare the results to the LSTM model results.

3. Tranlate the local XBboost model into a SageMaker XGBoost model and build, train, Hypertune and evaluate the results.



* Epocs: For this dataset and model, 200 epochs generally achieved the necessary convergence and was mostly used for all tests.
* Embedding: The size of the embedding layer was an important variable. However, pretraining proved to be unproductive. Since the results obtained match or exceeded the baseline, and Pretraining was far below, this approach was not tested extensively.
One note is that 50 dimensions were used here where 100 were used in the Baseline. For the small SEB dataset, this would probalby not be significant. The Riordan paper was testing various datasets.
* Flatening/Shaping: This layer had a surprisingly negative impact for this dataset. I would appear that due to the small size of the data set the added information provided by keeping the word vectors seperate may have helpedw converge during training.
* LSTM: A single LSTM layer seemed sufficient for this data but the size was important.
* Dropout: The dropout percentage was a very powerful way to fine-tune the results particulary to account for the overfitting during training.


In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation

**1. SEB Data with LSTM**

The evaulation begain by attempting to match the basline provided in the Riordan for variation on the LSTM/Embedding model. Their best results were accuracies in the low to mid 70's.
My approach was to first overfit to insure that training accuracy was at 100%. Once this was achieved, reduce the overfitting to try an imporove the test results. I also tested a variety of the major 
model adustments suggested in the Riorden paper as most effetive. On additional test was to use Flattening and Shaping which converts the Embedded dimension into a single vector from the vector per word representation.
This was recommended on some on-line resources as a possible improvement when using the Embedding layer.

Feature Results: 

* Epocs: For this dataset and model, 200 epochs generally achieved the necessary convergence and was mostly used for all tests.
* Embedding: The size of the embedding layer was an important variable. However, pretraining proved to be unproductive. Since the results obtained match or exceeded the baseline, and Pretraining was far below, this approach was not tested extensively.
One note is that 50 dimensions were used here where 100 were used in the Baseline. For the small SEB dataset, this would probalby not be significant. The Riordan paper was testing various datasets.
* Flatening/Shaping: This layer had a surprisingly negative impact for this dataset. I would appear that due to the small size of the data set the added information provided by keeping the word vectors seperate may have helpedw converge during training.
* LSTM: A single LSTM layer seemed sufficient for this data but the size was important.
* Dropout: The dropout percentage was a very powerful way to fine-tune the results particulary to account for the overfitting during training.

*Results Table:*

Each test is listed in the order they were performed with notes on the changes (also highlighted in the cell changed) and notes on the results. The best result is highlighted in green.

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/seb/seb_tf_train_results.png?raw=true)

**2. SEB Data with XGBoost**

Testing XGBoost both locally and on SageMaker is much more straight forward with no complexity in variations of the structure of the model. This is a straightforward hypertuning exercise.
The results after hypertuning actually matched or maybe were slightly better than all tests with LSTM. It was also far simpler to do. The big downside is that we don't learn much from the resutls about the underlying data and how variations to the model might be done to ultimately improve the restuls more significantly based on our learnings.
It was a good validation of the LSTM results but also highligted the big challenges that remain in predicting short answer data.



**3. SAG Data with LSTM**

**4. SAG Data with XGBoost**

**5.** **Ngrams:** SAG Data with XGBoost and added Ngram features. 

This leveraged the work previously done for sentiment analysis project
The answer is compared to the reference answer to generate ngrams. For this test ng1 and ng2 were determined to be the most useful. However, these two were found to only correlation to correct vs. incorrect answers at about 20%. That means given a student answer, about 30% of the correct/incorrect grade is predicted by the ng1 or ng2. Also ng1 and ng2 have a 72% correlation to each other.  

The results showed minimal gains over the encoded answers without ng1 and ng2. While not bad, **77.9%** was not sufficently better to warrant extensive testing of this approach at this time. I do believe that factoring in the reference answer compared to the student answer should add value but a different modeling approach may be needed to reap the advantages of these features.
 
    Correlation between Ngrams and Correct. (20 - 30%)
    Only 72%  correlated together.
    corr_matrix = ngrams.corr().round(2)
    display(corr_matrix)
                 ng1             2       correct
    ng1         1.00            0.72      0.29
    ng2         0.72            1.00      0.23
    correct     0.29            0.23      1.00
 
 The two histograms below show the values for ng1 and ng2 which range from 0 to 1 and how they compare for correct and incorrect answers. Ideally, the ngram values for correct answers would skew higher than the ngram values for the incorrect answers.
 This would show up in a histogram as a two hump view where the left hump was incorrect and the right is incorrect. What we see below is that except for very high values of ng1 and ng2, there is not a very strong corelation.
 What we see is some small signs of skew as expected for the 30% correlation but it's more like a downward sloap for ngram values from lower to higher in the same ratio as correct to incorrect.
 
 ![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag-Ngram-1-Histogram.png?raw=true)
 
 ![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag-Ngram-2-Histogram.png?raw=true) 
 
 
 
    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=1,
                  learning_rate=0.01, max_delta_step=0, max_depth=4,
                  min_child_weight=6, missing=None, n_estimators=10000, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=0.8, verbosity=0)
    Min preds: 0.0 max_preds: 1.0
    
    predictions   0.0  1.0
    actuals
    0.0           72  130
    1.0           32  499

    Recall:     0.940
    Precision:  0.793
    Accuracy:   0.779

In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization

One of the concerns with modeling is the distirbution of data. Specifically the ratio of correct to incorrect which also can be though of as the relative difficlty of the tests for it's students. 
Taken as a whole we showed that the 2/3 to 1/3 correct to incorrect did not seem to be a major factor in accuracy results.
In fact the SEB dataset had the reverse percentages to the SAG dataset but the results for accuracy were about the same.

The SAG dataset had many more questions, 87 vs. 4. This meant thatanalysis on a question level becomes possible and interesting. We have enough data to see if aspects of the questions impacted accuracy.
Looking at the quality of predictions both from an accuracy and precision point of view shows a strong correlation between the difficulty of the question and the accuracy of the prediction.
This could be interpreted as harder questions results in more random or inchorent answers which are harder to predict. This also impacts the quantity of correct results. When there are very few as compared to easier questions it's not surprising that predictability would differ with difficulty of the question. 


**Percentage of correct v.s. incorrect by Question**

The chart orders the question from lest percentage correct to most. This builds on the chart from the Exploritory Visualization section by adding predictions. 

![](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/results/Sag2-Prediction-Correct-Incorrect-Histogram.png?raw=true)

In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
