import json
import plotly
import pandas as pd


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib  # Updated import for joblib
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

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessagesCategories', engine)

# Load model
model = joblib.load("../models/classifier.pkl")

# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    # Genre counts for messages
    genre_counts = df.groupby('genre')['message'].count()
    genre_names = list(genre_counts.index)

    # Aid-related message counts by genre
    aid_rel1 = df[df['aid_related'] == 1].groupby('genre')['message'].count()
    aid_rel0 = df[df['aid_related'] == 0].groupby('genre')['message'].count()
    genre_names = list(aid_rel1.index)

    # Calculate the distribution of classes with 1
    class_distr1 = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum() / len(df)

    # Sort distribution values in descending order
    class_distr1 = class_distr1.sort_values(ascending=False)

    # Series of values that represent classes with 0
    class_distr0 = 1 - class_distr1
    class_names = list(class_distr1.index)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_rel1,
                    name='Aid related'
                ),
                Bar(
                    x=genre_names,
                    y=aid_rel0,
                    name='Aid not related'
                )
            ],
            'layout': {
                'title': 'Distribution of message by genre and \'aid related\' class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_distr1,
                    name='Class = 1'
                ),
                Bar(
                    x=class_names,
                    y=class_distr0,
                    name='Class = 0',
                    marker=dict(
                        color='rgb(212, 228, 247)'
                    )
                )
            ],
            'layout': {
                'title': 'Distribution of labels within classes',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Class",
                },
                'barmode': 'stack'
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()