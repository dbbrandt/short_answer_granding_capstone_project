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

Once the models are built and tested with the relatively small SEB data set, the entire set of tests are repeated with a larger more complex data, SAG2, which consists of 88 computer science questions. 
This data set was referened in the earlier Suzen paper (*2) which focused on text mining approaches to grading. This dataset was substantially larger and exhibited a wider set of the challenges to short answer grading.
Restuls from this combined with the SEB     

*Data Pre-Processing*: The first non-trivial task was to preprocess the raw datasets some of which are XML based and generate encoded train and test input datafiles.
On of the goals of the encondining was to build in the ability to decode the results once broken up and randomized into train and test sets. 
This decoding helps to better understand and visualize the challenges with some of the answers as compared to others. 

*Modeling*: The next step was building and training code for the LSTM XGBoost models were created.
A version for local testing was done first and then translated to work on SageMaker with Jupyter for Hypertuning. The LSTM models required custom coding to work in SageMaker.
 
*Tuning and Variations*: In addition, some variations and adjustments to the model were tested. These variations were docented in the Reordan paper as having varying impacts on success. 
The key ones tested in this project are:
* Pretrained v.s. Generated embeddings. 
  * For pretrained you can download datasets of embedding word vectors that reflect the relationshiop between words based on a much larger dataset.
  * Genreated embedding are specific to the current dataset and may have a more limited value for a small dataset.
* LSTM layers and tunning
* Dropout layers and flatening layers to reduce overfitting or improve fitting in general.

*Analysis and Results* The results of the various datasets, model approaches and tunning are analyzed and documented.

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

## II. Analysis
![](https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png)

_(approx. 2-4 pages)_

### Data Exploration

The datasets used for testing include two data primary sources. These are located in the /data/source_data directory.

**1**.	**SciEntsBank** (SEB) dataset. This data was taken from Dzikovska et al., 2012.  The SciEntsBank (SEB) dataset consists of science assessment questions and I will work the set with 2-way labels (correct/incorrect). *3 (Reordan)
  * The data is stored in XML format, one file for each of four questions.
  * Each file includes the questions text, correct answer text and a list of graded answers (correct/incorrect)
  * To make processing easier:
    * **question data** (question_id (0..3), question text, reference answer) were combined into a single file, **questions.csv**
    * **answer data** (question_id, answer text, score - 0/1)  were combined into a single file, **answers.csv**
  
  * Example Question Data
    * **Question:** Carrie wanted to find out which was harder, a penny or a nickel, so she did a scratch test. How would this tell her which is harder?
        * **Reference Response**: The harder coin will scratch the other.
        * **Correct**: The one that is harder will scratch the less harder one.
        * **incorrect**: She could tell which is harder by getting a rock and seeing if the penny or nickel would scratch it. Whichever one does is harder.

    * **Question:** A solution is a type of mixture. What makes it different from other mixtures?
        * **Reference Response**: A solution is a mixture formed when a solid dissolves in a liquid.
        * **Correct**: It dissolves the solid into a liquid that is see through
        * **incorrect**: A solution is a different type of mixture. Then they are a solution is a mixtures that dissolves.    


**2**. **Short Answer Grading**: University of North Texas short answer grading data set. 
These assignments/exams were assigned to an introductory computer science class. 
The student answers were collected via an online learning environment. 
The data set as a whole contains 80 questions and 2273 student answers. 
The answers were scored by two human judges, using marks between 0 (completely incorrect) and 5 (perfect answer). 
Data set creators treated the average grade of the two evaluators as the gold standard to examine the automatic scoring task. *1 (Suzen)

  * The data set is stored in a very complex to process format of multiple text files and sub-directories. 
  * Multiple versions of the same data is available in aggregated and file per question formats.
  * Score were not aggregated into a single file and had to be aggregated using a script.
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

 * Example Question Data
    * **Question:** Carrie wanted to find out which was harder, a penny or a nickel, so she did a scratch test. How would this tell her which is harder?
        * **Reference Response**: The harder coin will scratch the other.
        * **Correct**: The one that is harder will scratch the less harder one.
        * **incorrect**: She could tell which is harder by getting a rock and seeing if the penny or nickel would scratch it. Whichever one does is harder.

    * **Question:** A solution is a type of mixture. What makes it different from other mixtures?
        * **Reference Response**: A solution is a mixture formed when a solid dissolves in a liquid.
        * **Correct**: It dissolves the solid into a liquid that is see through
        * **incorrect**: A solution is a different type of mixture. Then they are a solution is a mixtures that dissolves.    


In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
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
