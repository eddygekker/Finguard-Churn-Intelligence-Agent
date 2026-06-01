import time
import pickle
import pandas as pd
from src.simulator import get_db_engine

MODEL_PATH = "models/random_forest_model.pkl"
ENCODER_PATH = "models/label_encoders.pkl"


def load_ml_artifacts():
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(ENCODER_PATH, 'rb') as encoder_file:
            encoders = pickle.load(encoder_file)
        return model, encoders
    except FileNotFoundError:
        print(f"[ERROR] Model or Encoder files not found at {MODEL_PATH}")
        return None, None


def preprocess_live_data(df, encoders):
    processed_df = df.copy()
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    for col in categorical_cols:
        if col in processed_df.columns and col in encoders:
            le = encoders[col]
            processed_df[col] = processed_df[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)

            if '<unknown>' not in le.classes_:
                import numpy as np
                le.classes_ = np.append(le.classes_, '<unknown>')

            processed_df[col] = le.transform(processed_df[col].astype(str))

    if 'CLIENTNUM' in processed_df.columns:
        processed_df = processed_df.drop(columns=['CLIENTNUM'])
    if 'churn_probability' in processed_df.columns:
        processed_df = processed_df.drop(columns=['churn_probability'])
    if 'retention_status' in processed_df.columns:
        processed_df = processed_df.drop(columns=['retention_status'])

    return processed_df


def run_inference_pipeline(check_interval_seconds=2):
    engine = get_db_engine()
    model, encoders = load_ml_artifacts()

    if not model or not encoders:
        print("[ERROR] Inference pipeline stopped due to missing ML artifacts.")
        return

    print("[INFERENCE] Inference pipeline is running and listening for 'New' customers...")

    feature_order = [
        'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
        'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ]

    while True:
        try:
            query = "SELECT * FROM bank_customers_stream WHERE retention_status = 'New'"
            new_customers_df = pd.read_sql(query, con=engine)

            if not new_customers_df.empty:
                print(f"\n[INFERENCE] Found {len(new_customers_df)} new customer profiles to process.")

                for _, row in new_customers_df.iterrows():
                    client_num = int(row['CLIENTNUM'])
                    single_row_df = pd.DataFrame([row])

                    processed_features = preprocess_live_data(single_row_df, encoders)
                    processed_features = processed_features[feature_order]

                    probabilities = model.predict_proba(processed_features)
                    churn_prob = float(probabilities[0][1])

                    next_status = 'Ignored'
                    if churn_prob >= 0.75:
                        next_status = 'Pending_AI'

                    from sqlalchemy import text
                    update_query = text(f"""
                        UPDATE bank_customers_stream 
                        SET churn_probability = {churn_prob}, retention_status = '{next_status}'
                        WHERE CLIENTNUM = {client_num}
                    """)

                    with engine.connect() as connection:
                        connection.execute(update_query)
                        connection.commit()

                    print(f"[INFERENCE] Client {client_num}: Churn Prob = {churn_prob:.2f} -> Status: {next_status}")

        except Exception as e:
            print(f"[ERROR] Inference loop encountered an error: {e}")

        time.sleep(check_interval_seconds)