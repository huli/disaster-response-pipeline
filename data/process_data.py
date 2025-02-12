import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:

    # Load messages dataset
    messages = pd.read_csv(messages_filepath)

    # Load categories
    categories = pd.read_csv(categories_filepath)

    # Merge datasets together
    df = messages.merge(categories, on=['id']) 

    return df

def clean_data(df) -> pd.DataFrame:
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.loc[0, :]

    # Use this row to extract a list of new column names
    category_colnames = [x[0] for x in row.str.split('-')]

    # Rename the columns of 'categories'
    categories.columns = category_colnames

    # Convert category values to 0 and 1
    # Example: aid_related-0 -> aid_related
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    
    # Drop the original categories column from 'df'
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new 'categories'
    df = pd.concat([df, categories], axis=1)

    # Drop duplicate rows
    df.drop_duplicates(subset=['message','original'], keep='first', inplace=True)

    # Assert the absence of duplicates
    assert sum(df.duplicated(['message', 'original'])) == 0

    return df

def save_data(df, database_filename):
    
    # Create sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Response', engine, index=False, if_exists='replace')  

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