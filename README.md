# Project Components
There are three components in this project.  

## 1. ETL Pipeline
In a Python script, process_data.py, wrote a data cleaning pipeline that:  

+ Loads the messages and categories datasets  
+ Merges the two datasets  
+ Cleans the data  
+ Stores it in a SQLite database
## 2. ML Pipeline
In a Python script, train_classifier.py, wrote a machine learning pipeline that:

+ Loads data from the SQLite database
+ Splits the dataset into training and test sets
+ Builds a text processing and machine learning pipeline
+ Trains and tunes a model using GridSearchCV
+ Outputs results on the test set
+ Exports the final model as a pickle file

## 3. Flask Web App
It is including:
+ Data visualizations using Plotly in the web app.
