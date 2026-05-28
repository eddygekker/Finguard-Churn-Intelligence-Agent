import os
import pandas as pd
from google import genai
from main import train_predictive_engine

# Dynamically resolve paths for the root configuration files
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
    - Calculated Churn Probability: {risk_prob:.2f}%
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


def run_retention_pipeline():
    print("🤖 Agent initializing: Training predictive engine...")
    model, label_encoders, y_encoder, X_test = train_predictive_engine()
    print("💥 Predictive engine trained and ready for inference.\n")

    secret_key = load_api_key()

    print("🔍 Agent checking customer profile for churn risk...")

    # Extract an actual high-risk customer profile sample
    high_risk_sample = X_test.iloc[0]

    # Calculate exact probability matrices
    probabilities = model.predict_proba([high_risk_sample])

    # Churn probability is usually index 0 depending on encoding, matching 'Attrited Customer'
    risk_score = probabilities[0][0] * 100

    print(f"📈 Analyzed Risk: {risk_score:.2f}% probability of churning.")

    # The Gatekeeper Business Rule Threshold (75%)
    if risk_score > 75:
        print("🚀 Risk threshold breached! Activating LLM Retention Agent...")
        trigger_llm_agent(high_risk_sample, risk_score, secret_key)
    else:
        print("✅ Risk levels nominal. No downstream agent interaction required.")


if __name__ == "__main__":
    run_retention_pipeline()