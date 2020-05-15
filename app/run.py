import json
import re
import warnings

import joblib
import nltk
import pandas as pd
import plotly
from flask import Flask
from flask import render_template, request, jsonify
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'stopwords', 'words'], quiet=True)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)


def tokenize(text):
    """
    :arg text: (string) text to process
    :return clean_tokens: (list) of lemmatized tokens
    """

    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']/df['message'].count()*100
    genre_names = list(genre_counts.index)

    cat_perc = df.drop(['message', 'original', 'genre'], axis=1).sum() / len(df) * 100
    cat_perc = cat_perc.sort_values()
    cat_names = cat_perc.index.to_list()

    # create visuals
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
                    'title': "Percentage [%]",
                    'showgrid': True
                },
                'xaxis': {
                    'title': "Genre"
                },
                'hoverinfo': 'y'
            }
        },

        {
            'data': [
                Bar(
                    x=cat_perc,
                    y=cat_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution by Categories',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Percentage [%]",
                    'showgrid': True
                },
                'hoverinfo': 'x'
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
