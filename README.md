### Disaster Pipeline for Twitter Messages
Dashboard for classification of messages sent during disaster



### Usage:
Below you find instructions on how to use the code

#### ETL Pipeline
The pipeline takes the messages.csv and categories.csv as input and produces combined sqlite table.

The command is as follows (when you are in /data):
* *python process_data.py messages.csv categories.csv DisasterResponse.db*

#### ML Pipeline
The pipeline takes the messages.csv and categories.csv as input and produces a model serialized as pickle file and 
* *python models/train_classifier.py data/DisasterResponses.db classifier.pkl*

#### Dashboard

* Run the following command in the app's directory to run your web app.
** python run.py**

#### Execute tests
The feature extraction is hardened with some tests. You can execute all tests by executing the following line in the root directory:

* *pytest*
