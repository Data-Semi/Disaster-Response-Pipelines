# Project Overview
I analyzed disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.  
The data set with csv files containing real messages that were sent during disaster events.  
I created a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.  

This project includes a web app where an emergency worker can input a new message and get classification results in several categories.    
The web app will also display visualizations of the data.   
Below are a few screenshots of the web app.  
Image #1:  
![Image1](https://github.com/Data-Semi/Disaster-Response-Pipelines/blob/master/images/image1.PNG)  

Image #2:  
![Image2](https://github.com/Data-Semi/Disaster-Response-Pipelines/blob/master/images/image2.PNG)  

Image #3:  
![Image3](https://github.com/Data-Semi/Disaster-Response-Pipelines/blob/master/images/image3.PNG)  

## Project Components
There are three components in this project.  

### 1. ETL Pipeline
In a Python script, process_data.py, wrote a data cleaning pipeline that:  

+ Loads the messages and categories datasets  
+ Merges the two datasets  
+ Cleans the data  
+ Stores it in a SQLite database  

Jupyter notebook link:[ETL Pipeline Preparation.ipynb](https://github.com/Data-Semi/Disaster-Response-Pipelines/blob/master/ETL%20Pipeline%20Preparation.ipynb) 
### 2. ML Pipeline
In a Python script, train_classifier.py, wrote a machine learning pipeline that:

+ Loads data from the SQLite database
+ Splits the dataset into training and test sets
+ Builds a text processing and machine learning pipeline
+ Trains and tunes a model using GridSearchCV
+ Outputs results on the test set
+ Exports the final model as a pickle file
    After compared weighted average of f1-scoref1-score in 3 models, 
    I chose RandomForest model and tuned with "class_weight" parameter in GridSearchCV.  
    
Jupyter notebook link:[ML Pipeline Preparation.ipynb](https://github.com/Data-Semi/Disaster-Response-Pipelines/blob/master/ML%20Pipeline%20Preparation.ipynb)

### 3. Flask Web App
It is including:
+ Data visualizations using Plotly in the web app.
All files are inside the folder pyFiles.Please check README file inside.  

Folder link:[pyFiles](https://github.com/Data-Semi/Disaster-Response-Pipelines/tree/master/pyFiles)  

## A brief insights on the project.  
+ Uninformative category data:     
 I found there is a label named "chiled-alone"in the original data only include 1 class.(value is all 0).   
 I droped the "chiled-alone" category from the original dataset in the first time.   
+ Imbalanced dataset:    
 This dataset is imbalanced as we can see in image #2 screenshort above.(ie some labels like water have few examples).   
 It will affects my training model's accuracy uninformative. For example, if my model predict all 0 in these labels which has few examples, the models accuracy willbe very high but it is not predicting well.     
  I tuned "class_weight" parameter in GridSearchCV in my RandomForest model.   
+ Emphasizing precision or recall for the various categories  
  I build 3 type of piplelines with GridSearchCV  
  Compare these 3 piplelines with weighted average of f1-score.  
    *Pipeline1 with MultinomialNB classifier which use naive bayes algorithm.  
    *Pipeline2 with RandomForestClassifier classifier wich use random frest algorithm.  
    *Pipeline3 with RandomForestClassifier classifier and an extra feature ContainsNumbers wich indicates whether a message includes a cardinal number or not.  
    Acording to line-chart comparison,I choose to export pipeline2 , because it has better result in most categories than others.  


