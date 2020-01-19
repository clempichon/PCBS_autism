# Machine learning on autism data

## Introduction 

 Pr Fadi Fayez Thabtah, from the Digital Technology Departement of Manukau Institute of Technology of Auckland has created an app, the Autism Spectrum Disorder Tests App, aiming at understanding the main characteristics of autistic patients in order to better detect ASD. By answering a series of 10 questions, and giving a few information about themselves, users of the app have contributed in creating datasets related to autism. It has been divided in three parts : children (from 4 to 11 years old), adolescents (12 to 18 years old), and adults. 

These datasets are the starting for this project of machine learning, aiming at creating an algorithm that detect autism from the responses to those 10 questions. 

I here present my strategy to build this algorithm. The code is disponible in the Jupyter Notebook "project.iynb". The data set are the three arff files available in the file. 

## 1. Importing the recquired modules

arff is important to load the arff files. I also load commmon modules such as pandas, numpy and matplotlib.pyplot. I then import several sklearn functions. Scikit-learn is a pre-defined package of Python made for machine learning.

## 2. Loading the data

The data is arff files, which contain two parts : one with metadata, and one with data. The first 10 columns are  responses to 10 different questions about their life habits. The results of the questions is summarized in the result variable, a score relative to the number of yes answers. 

<img src="photos/app.png" alt="app" width="600"/>

We also have data about the person : gender, ethnicity, jundice when he was young, country, age, whether one parent have a ASD-like trouble ('austim')... We also know if the child is autistic or not ('Class/ASD'). We also have a few information about the person responding the questionnaire (its relation to the child, its use of the app before etc.). 

The data set are loaded as data_child, data_ado and data_adult.

## 3. Handling missing values

I want to handle missing values by replacing it by their mean. I want to to this before merging the different data sets because I hypothesis that some variables are very different from a child to an adult. Therefore, I search for missing values and handle it before merging the datasets. Missing values are found in the age variable : I replace them by the mean of their category (mean age of children, mean age of adults).

## 4. Merging datasets

We have three different data sets (child, adolescent, adult). I merge them into a single one in order to have more observations and therefore better predictions. 

