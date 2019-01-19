import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from feature_extractor import StartingVerbExtractor, ResponseLengthExtractor, tokenize
import pickle

def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Response', engine)

    # Specify dependent and independent variables
    X = df.loc[:, ['message']]
    y = df.iloc[:, 5:] # related is not a category

    return X.message.values, y

def build_model():

    # Define complete pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('starting_verb', StartingVerbExtractor()),
            ('tweet_length', ResponseLengthExtractor()),
        ])),
        ('multi_clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, 
            n_jobs=-1, 
            n_estimators=10))),
    ])

    return pipeline

def evaluate_model(model, X_test, y_test):
    
    # Make prediction with learned parameters
    pred_grid = model.predict(X_test)

    # Print summary
    print_classification_summary(y_test, pred_grid)

def print_classification_summary(y_test, y_predicted):

    # collect the classification reports
    reports = []
    for index in range(len(y_test.columns)):
        reports.append(classification_report(y_test.values[:, index], y_predicted[:, index], output_dict=True))
        
    # create dataframe for cleaner printing
    results = pd.DataFrame(
        {'micro avg': [report['micro avg']['f1-score'] for report in  reports],
         'macro avg': [report['macro avg']['f1-score'] for report in  reports], 
         'weighted avg': [report['weighted avg']['f1-score'] for report in  reports]}
    )

    # add total column
    results = results.append(results.sum() / len(results), ignore_index=True)

    # add category column
    results = pd.concat([pd.DataFrame({'category': y_test.columns.append(pd.Index(['total'])).values}), 
                              results], axis=1, sort=False)

    # print results
    print(results)
    print('Average f1-score over all categories: %s (weighted avg)' % results.iloc[-1, 3])


def save_model(model, model_filepath):
    
    # Save the model to disk
    pickle.dump(model, open('model.pkl', 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()