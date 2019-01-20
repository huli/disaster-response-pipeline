## Disaster Response Project
Dashboard for classification of messages sent during disaster


* [Project Description](#project-description) 
* [Getting started](#getting-started)
  * [Installing Requirements](#installing-requirements)
  * [Cleaning data](#cleaning-data)
  * [Training model](#training-model)
  * [Starting dashboard](#starting-the-dashboard)
  * [Tests](#tests)
* [Additional material](#additional-material)
  * [Authors](#authors)   
  * [Licence](#licence)   
  * [Screenshot](#sccreenshot)
  * [Acknowledgments](#acknowledgments)
  
  

### Project Description
The target of this project is to classify messages sent during and after a disaster. With a correct classification of the messages the different disaster response organizations are able to provide much more efficient and effective help for the people that need it. 


This project is part of the Data Science Nanodegree [@Udacity](https://www.udacity.com) and is done in collaboration with [FigureEight](https://www.figure-eight.com/).

There are three main parts of the project:

1. **ETL Pipeline** - takes the messages.csv and categories.csv as input and produces combined sqlite table.
2. **ML Pipeline** - trains and evaluates the model and produces a pickle file from the trained model. This model is then used from the dashboard.
3. **Dashboard** - shows some statistics about the training data and allows the user to input and classify a message.


### Getting started

#### Installing Requirements

The following packages need to be installed for the project:

* numpy, pandas
* scikit-learn
* nltk
* sqlalchemy, pickle
* flask, plotly, json
* pytest

#### Cleaning data

To run the etl pipeline you can execute the following command in the data folder:
````
python process_data.py messages.csv categories.csv DisasterResponse.db
````
#### Training the model

To train and serialize the model you can execute the following command:
````
python models/train_classifier.py data/DisasterResponses.db classifier.pkl
````

#### Starting the Dashboard

To start the web app you can execute this line in the root directory:
````
python run.py
````

#### Execute tests
The feature extraction is hardened with some tests. You can execute all tests by executing the following line in the root directory:

* *pytest*

#### Screenshot

![Sample Input](images/response_input.png)