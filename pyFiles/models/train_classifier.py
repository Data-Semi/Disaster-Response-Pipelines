import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import pickle
import warnings
import time

import matplotlib.pyplot as plt


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('1st',engine)
    X = df.message.values
    df_Y = df.drop(["message","id","original","genre"],axis = 1)
    #The category "child_alone" only has 1 class. I decide to remove this column because it is not informative for machine learning.
    df_Y = df_Y.drop("child_alone",axis = 1)
    Y = df_Y.values
    category_names = df_Y.columns
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__class_weight':("balanced",None) 
     }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col+":")
        print(classification_report(Y_test[i], Y_pred[i]))


def save_model(model, model_filepath):
    with open(model_filepath, mode='wb') as fp:
        pickle.dump(model, fp)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        warnings.simplefilter('ignore', FutureWarning)
        start_time = time.time()
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        elapsed_time = time.time()-start_time
        print("elapsed_time:{}".format(elapsed_time))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()