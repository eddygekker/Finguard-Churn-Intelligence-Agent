import os
import time
import pandas as pd
from google import genai
from sqlalchemy import text
from src.simulator import get_db_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEY_PATH = os.path.join(BASE_DIR, 'api_key.txt')


def load_api_key():
    if not os.path.exists(API_KEY_PATH):
        raise FileNotFoundError(f"Critical Error: 'api_key.txt' missing from root directory: {API_KEY_PATH}")
    with open(API_KEY_PATH, 'r') as file:
        return file.read().strip()


def trigger_llm_agent(customer_profile, risk_prob, api_key):
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an elite Fintech Customer Retention & Risk Intelligence Agent at FinGuard Banking.
    A high-value customer has been flagged by our Predictive Machine Learning Engine with a high risk of churning.

    CRITICAL INPUT METRICS:
    - Calculated Churn Probability: {risk_prob * 100:.2f}%
    - Customer Account Age (Months on book): {customer_profile['Months_on_book']} months
    - Total Inactive Months (Last 12 Months): {customer_profile['Months_Inactive_12_mon']} months
    - Credit Limit: ${customer_profile['Credit_Limit']}
    - Total Revolving Balance: ${customer_profile['Total_Revolving_Bal']}
    - Total Transaction Amount (Last 12 Months): ${customer_profile['Total_Trans_Amt']}
    - Total Transaction Count (Last 12 Months): {customer_profile['Total_Trans_Ct']}
    - Utilization Ratio: {customer_profile['Avg_Utilization_Ratio']}

    YOUR TASK:
    Generate a professional executive brief entirely in English consisting of two distinct sections:

    1. INTERNAL FINANCIAL RISK ANALYSIS:
       An analytical breakdown of why this specific customer is leaving, referencing their specific transaction counts, revolving balance, and utilization ratio. State the potential financial impact to the bank.

    2. TARGETED RETENTION MECHANICS & CUSTOMER OUTREACH:
       Provide a structured bullet-point strategy of custom financial incentives tailored to this profile (e.g., specific fee waivers or tailored credit extensions based on their credit limit). Following the strategy, draft a highly personalized, empathetic, and sophisticated outreach email to the customer, signed as 'FinGuard Executive Retention Team'.

    Tone: Highly analytical, corporate, and proactive. Do not include any Hebrew text.
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    print("\n========================= AGENT INTELLIGENCE REPORT =========================\n")
    print(response.text)
    print("\n=============================================================================\n")


def run_retention_pipeline(check_interval_seconds=5):
    engine = get_db_engine()

    try:
        api_key = load_api_key()
    except Exception as e:
        print(f"[ERROR] Agent stopped: {e}")
        return

    print("[AGENT] Retention Agent pipeline is running and listening for 'Pending_AI' customers...")

    while True:
        try:
            query = "SELECT * FROM bank_customers_stream WHERE retention_status = 'Pending_AI'"
            alert_customers_df = pd.read_sql(query, con=engine)

            if not alert_customers_df.empty:
                print(f"\n[AGENT] Detected {len(alert_customers_df)} high-risk profiles breach threshold!")

                for _, row in alert_customers_df.iterrows():
                    client_num = int(row['CLIENTNUM'])
                    risk_score = float(row['churn_probability'])

                    print(f"[AGENT] Activating LLM Generation for Client {client_num} (Risk: {risk_score * 100:.1f}%)")

                    customer_profile = row.to_dict()
                    trigger_llm_agent(customer_profile, risk_score, api_key)

                    update_query = text(f"""
                        UPDATE bank_customers_stream 
                        SET retention_status = 'Processed_By_Agent'
                        WHERE CLIENTNUM = {client_num}
                    """)

                    with engine.connect() as connection:
                        connection.execute(update_query)

                    print(f"[DATABASE] Client {client_num} updated to 'Processed_By_Agent'")

        except Exception as e:
            print(f"[ERROR] Agent pipeline encountered an error: {e}")

        time.sleep(check_interval_seconds)