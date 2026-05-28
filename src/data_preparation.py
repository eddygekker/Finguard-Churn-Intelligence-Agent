import os
import pandas as pd
from sqlalchemy import create_instance

# Dynamically resolve paths based on project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BankChurners.csv')
OUTPUT_DATA_PATH = os.path.join(BASE_DIR, 'data', 'finguard_data.csv')


def prepare_and_load_data():
    print("Starting data preparation pipeline...")

    # Load raw dataset from data folder
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Source file not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Clean column names (remove suffixes if exist)
    df = df.loc[:, ~df.columns.str.contains('Naive_Bayes')]

    # Save cleaned dataset back to data folder
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"Cleaned data saved locally at: {OUTPUT_DATA_PATH}")

    # Establish connection and load to MySQL Warehouse
    engine = create_instance("mysql+pymysql://root:password@localhost/finguard_db")
    df.to_sql('customer_churn_records', con=engine, if_exists='replace', index=False)
    print("Successfully synchronized Data Warehouse table: customer_churn_records")


if __name__ == "__main__":
    prepare_and_load_data()