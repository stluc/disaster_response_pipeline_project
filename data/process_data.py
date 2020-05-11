import sys

import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    """Receives the paths to two csv files and transforms them into a common Pandas DataFrames.
        :param messages_filepath: (string) Path to the messages csv.
        :param categories_filepath: (string) Path to the categories csv.
        
        :return: df - Pandas DataFrame merged from the two input csv.
        :rtype: Pandas DataFrame
    """
    df: DataFrame = pd.DataFrame()
    try:
        assert isinstance(messages_filepath, str), 'Messages filepath should be a string.'
        messages = pd.read_csv(messages_filepath)
        assert isinstance(categories_filepath, str), 'Categories filepath should be a string.'
        categories = pd.read_csv(categories_filepath)

        df = messages.merge(categories, how='left').drop('id', axis=1, errors='ignore')

    except (AssertionError, TypeError) as error:
        print(error)
    except FileNotFoundError as error:
        print(error)
    except:
        print('Unknown error while loading the data.')

    finally:
        return df


def clean_data(df):
    """Receives the Pandas DataFrame and cleans up the categories labels, sorting them into separate
    columns.
        :param: df: (DataFrame) Data coming from the load_data() method.

        :return: clean_df: (DataFrame) Categories rearranged into columns and duplicate data removed.
    """
    clean_df: DataFrame = df.drop('categories', axis=1).copy()
    try:
        categories = df.categories.str.split(';', expand=True)
        row = categories.iloc[0]
        categories.columns = [cat.partition('-')[0] for cat in row]
        for column in categories:
            # set each value to be the last character of the string
            categories[column] = categories[column].astype(str).str[-1]

            # convert column from string to numeric
            categories[column] = pd.to_numeric(categories[column])

        clean_df = pd.concat([df, categories], axis=1).drop_duplicates()

    except:
        print('Unknown error while cleaning the data.')

    finally:
        return clean_df


def save_data(df, database_filename):
    """
        :arg df: (DataFrame) Pandas DataFrame to be saved
        :arg database_filename: (string) Name of the SQLite Database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
