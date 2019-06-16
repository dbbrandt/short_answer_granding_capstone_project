           
 	 	           Automatic Short-Answer Grading
                        Questions/Student Answers with Grades
                          ==========================
                            
                                 Version 1.0
                                   
                              October 29th, 2008

                         Rada Mihalcea, Michael Mohler

                      Language and Information Technologies
                         University of North Texas

                               rada@cs.unt.edu
                               mgm0038@unt.edu
  

CONTENTS
1. Introduction
2. Data Set
2.a. Data Annotation
2.b. Folder Structure
3. Feedback
4. Citation Info
5. Acknowledgments




=======================
1. Introduction

This README v1.0 (October, 2008) for the automatic short-answer grading 
data set comes from the archive hosted at the following URL
http://lit.csci.unt.edu/index.php/Downloads

=======================
2. Data Set

=====
2.a. Data Annotation

Our data set consists of three assignments of seven questions each given to
an introductory computer science class at the University of North Texas. The
data is in plaintext format. Each assignment includes the question, teacher 
answer, and set of student answers with the average grades of two annotators
included. Both annotators were asked to grade for correctness on an integer
scale from 0 to 5. The inter-annoatator correlation on the data set was .6443
using Pearson's coefficient. A unique student ID is also provided.

The format for the assignment files is as follows. For each question in the
assignment, the first 4 lines are, respectively, a sentinal line containing
only '#' characters, the question line beginning with the string
"Question: ", the answer line beginning with the string "Answer: ", and a
blank line. The next N lines (where N is the number of students who
submitted the assignment) each contain the average grade given for the
answer, a unique student id associated with the answer (in brackets []), and
finally the raw answer itself. Finally, after all the answer lines, a blank
line preceeds the start of the next question.

Consider the graphic below with line numbers and <> for	reference:

		1 | #################################
		2 | 	Question: <QUESTION1>
		3 |	Answer: <ANSWER1>
		4 |
		5 | <Grade1:1> [<Student1>] <StudentAnswer1:1>
		6 | <Grade1:2> [<Student2>] <StudentAnswer1:2>
		               ...
		32| <Grade1:28> [<Student28>] <StudentAnswer1:28>
		33|
		34| #################################
		35|	Question: <QUESTION2>
		36|     Answer: <ANSWER2>
		37|
		38| <Grade2:1> [<Student1>] <StudentAnswer2:1>
				...   

For privacy reasons, no actual student identifiers are used in this corpus.
Instead, each student has been assigned an integer from 0 to 30 for use
in the uid files.


=====
2.b. Folder Structure

*StudentAnswers
	This folder contains three assignments of seven questions each in 
	plaintext format.

     Files:
	assign1.txt 
	assign2.txt
	assign3.txt - The assignment files as described above for
		assignments 1, 2, and 3, respectively.



=======================
3. Feedback

For further questions or inquiries about this data set, you can contact:
Michael Mohler (mgm0038@unt.edu) or Rada Mihalcea (rada@cs.unt.edu).


=======================
4. Citation Info 

If you use this data set, please cite:

@InProceedings{Mohler08b,
  author =       {Michael Mohler, Rada Mihalcea},
  title =        {Text-to-text Semantic Similarity for Automatic Short Answer Grading},
  booktitle =    {Proceedings of the Conference of the European Association of Computational Linguistics},
  address =      {Athens, Greece},
  year =         {2009}
}

=======================
5. Acknowledgments

This work was partially supported by a National Science Foundation grant IIS-
#0747340. Any opinions, findings, conclusions or recommendations expressed
above are those of the authors and do not necessarily reflect the views of the
National Science Foundation.


