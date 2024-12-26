import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.simplefilter('ignore')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """Load and merge messages and categories datasets from the database."""
    logging.info("Loading data from database.")
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessagesCategories', con=engine)  # Ensure this matches your database table name
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # Convert all category columns to integers
    Y = Y.astype(int)
    
    category_names = Y.columns.tolist()
    
    logging.info(f"Loaded {X.shape[0]} messages and {Y.shape[1]} categories.")
    return X, Y, category_names

def tokenize(text):
    """Normalize, tokenize, and lemmatize text string."""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, min_df=5)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    
    parameters = {
        'vect__min_df': [1, 5],
        'tfidf__use_idf': [True, False],
        'clf__estimator__min_samples_split': [2]
    }

    scorer = make_scorer(f1_score, average='macro')  # Use built-in f1_score
    return GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=2, n_jobs=-1)

def get_eval_metrics(y_true, y_pred, col_names):
    """Calculate evaluation metrics for each category."""
    metrics = []
    for i, col in enumerate(col_names):
        try:
            precision = precision_score(y_true.iloc[:, i], y_pred[:, i])
        except ZeroDivisionError:
            precision = 0  # Handle zero division case
        
        metrics.append([
            accuracy_score(y_true.iloc[:, i], y_pred[:, i]),
            precision,
            recall_score(y_true.iloc[:, i], y_pred[:, i]),
            f1_score(y_true.iloc[:, i], y_pred[:, i])
        ])
    return pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'F1'], index=col_names)

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model performance and print metrics."""
    logging.info("Evaluating model.")
    Y_pred = model.predict(X_test)
    metrics_df = get_eval_metrics(Y_test, Y_pred, category_names)
    logging.info("\n" + str(metrics_df))

def save_model(model, model_filepath):
    """Save the model as a pickle file."""
    logging.info(f"Saving model to {model_filepath}.")
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        logging.info(f"Database filepath: {database_filepath}")

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        logging.info("Building model.")
        model = build_model()
        
        logging.info("Training model.")
        model.fit(X_train, Y_train)
        
        evaluate_model(model, X_test, Y_test, category_names)
        
        save_model(model, model_filepath)
        logging.info("Model saved successfully!")
    else:
        logging.error("Please provide the correct arguments: database filepath and model filepath.")
        print("Usage: train_classifier.py ../data/DisasterResponse.db classifier.pkl")

if __name__ == '__main__':
    main()