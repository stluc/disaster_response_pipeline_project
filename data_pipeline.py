# import packages
import sys
import os
import re
from contextlib import redirect_stdout

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

with redirect_stdout(open(os.devnull, "w")):
    nltk.download(['punkt', 'wordnet', 'stopwords', 'words',
                   'averaged_perceptron_tagger', 'maxent_ne_chunker', 'omw'])


def load_data(data_file):
    """
    :arg data_file: (string) file path to SQLite Database
    containing DisasterResponse table.
    :return x: (Array) variable columns
    :return y: (Array) classifications
    :return labels: (list) column label for y
    """
    engine = create_engine('sqlite:///' + data_file)
    df = pd.read_sql_table('DisasterResponse', engine)
    x = df.message.values
    y = df.drop(['message', 'original', 'genre'], axis=1, errors='ignore')
    labels = y.columns
    y = y.values

    return x, y, labels


def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def tokenize(text):
    """
    :arg text: (string) text to process
    :return clean_tokens: (list) of lemmatized tokens
    """

    # Normalize text
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    tokens = nltk.pos_tag(tokens)

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok, tag in tokens:
        clean_tok = lemmatizer.lemmatize(tok.strip(), pos=get_wordnet_pos(tag))
        clean_tokens.append(clean_tok)

    return clean_tokens


def display_results(y_pred, y_test, labels):
    """ Get a classification report for each output and prints out the weighted f1 score.
    :arg y_pred: (array) of predicted values
    :arg y_test: (array) of expected values
    :param labels: (list) of labels for y outputs

    :return cr: (dict) of multioutput classification reports where keys are the labels of y
    """
    cr = {}
    model_avg_f1 = np.empty(len(labels))
    for i, label in enumerate(labels):
        cr[label] = classification_report(
            y_test[:, i], y_pred[:, i], labels=df[label].unique(), zero_division=0)
        score = f1_score(y_test[:, i], y_pred[:, i], labels=df[label],
                         average='weighted', zero_division=0)
        model_avg_f1[i] = score

    model_avg_f1 = np.mean(model_avg_f1)
    print(f'The model weighted f1 score is {model_avg_f1}')
    return cr


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    parameters = {
        'tfidf__use_idf': [True, False],
        'tfidf__smooth_idf': [True, False]
    }

    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1)

    return model_pipeline


def train(x, y, model, labels):

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cr = display_results(y_pred, y_test, labels)

    return model, cr


def export_model(model):
    joblib.dump(model, 'model')


def run_pipeline(data_file):

    x, y, labels = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model, cr = train(x, y, model, labels)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
