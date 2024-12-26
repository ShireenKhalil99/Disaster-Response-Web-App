#importing necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = messages.merge(categories, on='id', how='inner', validate='many_to_many')

    # Split 'categories' into separate columns
    categories_split = df['categories'].str.split(';', expand=True)

    # Extract column names from the first row
    row = categories_split.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename columns
    categories_split.columns = category_colnames

    # Convert category values to numbers
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1]
        categories_split[column] = pd.to_numeric(categories_split[column], errors='coerce')

    # Drop the original 'categories' column
    df.drop(columns=['categories'], inplace=True)

    # Concatenate with the new categories dataframe
    df = pd.concat([df, categories_split], axis=1, sort=False)

    return df

def clean_data(df):
    # Drop duplicates
    df2 = df.drop_duplicates(subset=['message'])

    # Drop 'child_alone'
    df3 = df2.drop('child_alone', axis=1)

    # Filter out rows where 'related' == 2
    df4 = df3[df3.related != 2]

    return df4

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + str(database_filepath))
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
