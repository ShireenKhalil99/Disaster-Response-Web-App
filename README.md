# Disaster-Response-Web-App 💻

## Installation 💡
This repository houses code developed using HTML and Python 3. To successfully execute this code, you must install the following Python libraries: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle, and warnings.

## Project Motivation ⁉️ 
> In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

## File Descriptions ✍️
1. ETL Pipeline (In a Python script, **process_data.py**)
   - Loads the messages and categories datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database
     
2. ML Pipeline (In a Python script, **train_classifier.py**)
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

3. Flask Web App

   - Modify file paths for database and model as needed
   - Add data visualizations using Plotly in the web app. One example is provided for you

### Instructions 📙
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db  models/classifier.pkl`
        
2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


## Result ✔️

- Upon successful execution, the code is expected to generate output visually resembling the provided screenshots.

<img width="1000" alt="OverviweTraining-set" src="https://github.com/user-attachments/assets/0b7c9d68-8bca-402e-8283-5c96cb6cee4d" />

<img width="1071" alt="Distrbuation" src="https://github.com/user-attachments/assets/9a53a461-88ef-48c4-8471-d5b603a7de04" />

<img width="1060" alt="Disaster Response Project" src="https://github.com/user-attachments/assets/6e837ddd-9404-41b0-8e2a-4aa2ef429400" />

## Licensing, Authors, Acknowledgements ♥️

This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Udacity reserves all the rights of materials for this project.
