import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('1st', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_genre_scatter = df.groupby("genre").sum().drop(["id"],axis = 1)
    genre_names = list(df_genre_scatter.index)
    first_gen = df_genre_scatter[:1]
    second_gen = df_genre_scatter[1:2]
    third_gen = df_genre_scatter[2:3]
    

    df_cate_hist = df.drop(["message","id","original","genre"],axis = 1)
    df_cate_hist["count_cate"] = df_cate_hist.sum(axis = 1)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=list(first_gen.columns),
                    y=first_gen.values[0],
                    marker={'color': 'blue', 'symbol': 'star', 'size': 10},
                    mode='markers',
                    name=first_gen.index[0]
                ),
                Scatter(
                    x=list(second_gen.columns),
                    y=second_gen.values[0],
                    marker={'color': 'red', 'symbol': 'circle', 'size': 5},
                    mode='markers',
                    name=second_gen.index[0]
                ),
                Scatter(
                    x=list(third_gen.columns),
                    y=third_gen.values[0],
                    marker={'color': 'green', 'symbol': 'x', 'size': 5},
                    mode='markers',
                    name=third_gen.index[0]
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories By Genre',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Categories"
                },
                'font': {
                    'size': 8
                }
            }
        },
#        Histogram(x=x)
        {
            'data': [
                Histogram(
                    x=df_cate_hist["count_cate"]
                )
            ],

            'layout': {
                'title': 'Histogram In Matched Genres Number of Every Message',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Matched Genres Number"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()