# Disaster-Response-Web-App 💻

## Installation 💡
This repository houses code developed using HTML and Python 3. To successfully execute this code, you must install the following Python libraries:
1. json
2. plotly
3. pandas
4. nltk
5. flask
6. sklearn
7. sqlalchemy
8. sys
9. numpy
10. re
11. pickle
12. warnings


## Project Motivation ⁉️ 
This repository hosts the code for a web app designed to assist emergency workers during disaster events like earthquakes or hurricanes. The app classifies disaster messages into predefined categories, ensuring that each message is routed to the appropriate aid agency. 

At its core, the app leverages a machine learning model to automatically categorize incoming messages. The repository also includes the codebase for training the model and preprocessing new datasets for future training.

### File and Folder Descriptions

1. **`process_data.py`**  
   This script processes input CSV files containing message data and message category labels. It merges and cleans the data before storing it in an SQLite database.

2. **`train_classifier.py`**  
   This script uses the SQLite database generated by `process_data.py` to train and optimize a machine-learning model for message categorization. The output includes:  
   - A pickle file containing the trained model.  
   - Test evaluation metrics printed during the training process.

3. **`data/`**  
   A folder containing sample CSV files for message and category datasets is used as input for the data processing pipeline.

4. **`app/`**  
   This folder contains all the necessary files for running and rendering the web application.
   

### Steps to Run the Project📙

1. Run process_data.py
   
- Place the data folder in the current working directory. Inside the data folder, save the process_data.py script.
  From the current working directory, execute the following command:
  
   **python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**

2. Run train_classifier.py
   
- In the current working directory, create a folder named models and save the train_classifier.py script inside this folder.
  From the current working directory, execute the following command:
  
  **python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**
  
3. Run the Web App
   
- Save the app folder in the current working directory.
  Navigate to the app directory and run the following command:
  python run.py
  Open a browser and go to http://0.0.0.0:3001/ to access the web app.


## Results✔️

- Upon successful execution, the code is expected to generate output visually resembling the provided screenshots.

<img width="1000" alt="OverviweTraining-set" src="https://github.com/user-attachments/assets/0b7c9d68-8bca-402e-8283-5c96cb6cee4d" />

<img width="1071" alt="Distrbuation" src="https://github.com/user-attachments/assets/9a53a461-88ef-48c4-8471-d5b603a7de04" />

<img width="1060" alt="Disaster Response Project" src="https://github.com/user-attachments/assets/6e837ddd-9404-41b0-8e2a-4aa2ef429400" />

## Warnings

- The datasets in this repository are highly imbalanced, with several message categories having very few positive examples. In some cases, positive examples make up less than 5% or even less than 
  1% of the data. This imbalance can lead to a classifier that achieves high accuracy by predominantly predicting that messages do not belong to these categories. However, this comes at the cost of 
  low recall, meaning a significant proportion of positive examples may be mislabeled. Therefore, caution is advised when relying on the app's results for decision-making purposes.

## Licensing, Authors, Acknowledgements ♥️

This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Udacity reserves all the rights of materials for this project.
