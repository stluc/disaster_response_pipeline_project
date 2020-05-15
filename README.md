# Disaster Response Pipeline Project
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The aim of the project is to categorize messages from different possible sources and provide useful information 
for Disaster Response teams.

This was achieved through an ETL Pipeline which transforms the datasets provided by Figure Eight and loads it
into a database, which will be used for training the classifier.
A Web App is provided (only on localhost).

## Table of Contents
1. [Dependencies](#dependencies)
2. [How to use](#usage)
3. [Extra](#extra)
4. [Acknowledgements](#acknowledgements)

<a name="dependencies"></a>
## Dependencies
The dependencies are listed in requirements.txt

These mainly consist of:

* Python 3.7
* NumPy, Pandas, Scikit-Learn
* NLTK
* SQLalchemy
* Flask, Bootstrap, JQuery, Plotly

<a name="usage"></a>
## How to Use
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/


<a name="extra"></a>
### Extra
Three exploratory Jupyter notebooks are provided in the root folder. These can be consulted
to verify the different approaches undertaken or explore new ones.
+ ETL Pipeline Preparation - Everything related to the dataset processing and SQL database creation
+ ML Pipeline Preparation - Everything related to the Machine Learning model, classifier evaluation and parameters tuning
+ Visualizations - Some experimenting with Plotly libraries 


<a name="acknowledgement"></a>
## Acknowledgements
* [Udacity](https://www.udacity.com/)
* [Figure Eight](https://www.figure-eight.com/)
