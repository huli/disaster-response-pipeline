import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from models.feature_extractor import tokenize, StartingVerbExtractor, ResponseLengthExtractor

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponses.db')
df = pd.read_sql_table('Response', engine)

# load model
model = joblib.load("models/model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    without_related = df.drop('related', axis=1)
    category_names = without_related.iloc[:,4:].columns
    category_counts = (without_related.iloc[:,4:] != 0).sum().values

    category_counts_sorted = sorted(zip(category_names, category_counts), key=lambda x: x[1])

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
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=[x[1] for x in category_counts_sorted],
                    y=[x[0] for x in category_counts_sorted],
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'xaxis': {
                    'title': "Count"
                },
                'yaxis': {
                    'title': ""
                },
                'height': 800,
                'margin' : {
                    'l' : 150
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
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()