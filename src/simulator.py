import time
import json
import random
import pymysql
import pandas as pd
from sqlalchemy import create_engine, text

DB_USER = "root"
DB_PASSWORD = "Gekker1999"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "finguard_db"


def get_db_engine():
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)


def create_stream_table():
    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS bank_customers_stream (
                CLIENTNUM INT PRIMARY KEY,
                Customer_Age INT,
                Gender VARCHAR(5),
                Dependent_count INT,
                Education_Level VARCHAR(20),
                Marital_Status VARCHAR(20),
                Income_Category VARCHAR(20),
                Card_Category VARCHAR(20),
                Months_on_book INT,
                Total_Relationship_Count INT,
                Months_Inactive_12_mon INT,
                Contacts_Count_12_mon INT,
                Credit_Limit DOUBLE,
                Total_Revolving_Bal INT,
                Avg_Open_To_Buy DOUBLE,
                Total_Amt_Chng_Q4_Q1 DOUBLE,
                Total_Trans_Amt INT,
                Total_Trans_Ct INT,
                Total_Ct_Chng_Q4_Q1 DOUBLE,
                Avg_Utilization_Ratio DOUBLE,
                churn_probability DOUBLE DEFAULT NULL,
                retention_status VARCHAR(30) DEFAULT 'New'
            );
            """
            cursor.execute(create_table_query)
        connection.commit()
        print("[DATABASE] Table 'bank_customers_stream' is ready.")
    finally:
        connection.close()


def load_and_clean_base_data(csv_path="data/BankChurners.csv"):
    print(f"[SIMULATOR] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    columns_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    if 'Attrition_Flag' in df.columns:
        df = df.drop(columns=['Attrition_Flag'])

    return df


def start_streaming(df, interval_seconds=3):
    engine = get_db_engine()
    print(f"[SIMULATOR] Starting live stream. Interval: {interval_seconds} seconds.")

    while True:
        random_index = random.randint(0, len(df) - 1)
        customer_row = df.iloc[random_index]

        customer_dict = customer_row.to_dict()
        customer_json = json.dumps(customer_dict)

        client_num = customer_dict['CLIENTNUM']
        print(f"\n[STREAM EVENT] New profile update arrived for Client: {client_num}")

        single_customer_df = pd.DataFrame([customer_dict])
        single_customer_df['retention_status'] = 'New'
        single_customer_df['churn_probability'] = None

        try:
            with engine.connect() as connection:
                connection.execute(text(f"DELETE FROM bank_customers_stream WHERE CLIENTNUM = {client_num}"))

            single_customer_df.to_sql(
                name='bank_customers_stream',
                con=engine,
                if_exists='append',
                index=False
            )
            print(f"[DATABASE] Successfully ingested client {client_num} with status 'New'")

        except Exception as e:
            print(f"[ERROR] Failed to ingest client {client_num}: {e}")

        time.sleep(interval_seconds)