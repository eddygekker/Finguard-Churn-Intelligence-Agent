import pandas as pd
from google import genai
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier

# Connection configuration
DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'finguard_db'
DB_PASSWORD = 'Gekker1999'


def load_api_key():
    """
    Reads the Gemini API key securely from a local text file.
    """
    try:
        # FIXED: Changed 'current_user_r' to correct 'r' mode
        with open("api_key.txt", "r") as file:
            key = file.read().strip()
            return key
    except FileNotFoundError:
        print("❌ Error: api_key.txt file not found. Please create it in the project root.")
        return None


def load_and_train_model():
    """
    Connects to the database, preprocesses the data,
    and trains the predictive Random Forest model.
    """
    print("🤖 Agent initializing: Training predictive engine...")
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(connection_string)

    query = "SELECT * FROM raw_customers;"
    df = pd.read_sql(query, engine)

    if 'CLIENTNUM' in df.columns:
        df = df.drop(columns=['CLIENTNUM'])

    target_map = {'Existing Customer': 0, 'Attrited Customer': 1}
    df['Attrition_Flag'] = df['Attrition_Flag'].map(target_map)

    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=['Attrition_Flag'])
    y = df_encoded['Attrition_Flag']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("💥 Predictive engine trained and ready for inference.")

    return model, X.columns


def run_agent_inference(model, feature_columns, sample_customer_data):
    """
    Simulates the agent checking a specific customer's risk profile.
    """
    print("\n🔍 Agent checking customer profile for churn risk...")

    customer_df = pd.DataFrame([sample_customer_data])
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    customer_encoded = pd.get_dummies(customer_df, columns=categorical_cols)
    customer_final = customer_encoded.reindex(columns=feature_columns, fill_value=0)

    probabilities = model.predict_proba(customer_final)[0]
    churn_probability = probabilities[1]

    print(f"📈 Analyzed Risk: {churn_probability * 100:.2f}% probability of churning.")
    return churn_probability


def trigger_llm_agent(customer_data, risk_score, api_key):
    """
    Triggers the cognitive generative AI layer using the modern google-genai SDK
    to build a personalized retention plan and email message.
    """
    print("🚀 Risk threshold breached! Activating LLM Retention Agent...")

    # Initialize the modern GenAI Client with the API key
    client = genai.Client(api_key=api_key)

    # Define the cognitive system personality instructions
    system_instruction = (
        "You are a senior customer retention specialist at FinGuard fintech bank. "
        "Your goal is to retain high-risk customers by analyzing their behavior traits "
        "and offering them highly targeted, attractive personal benefits. "
        "Be professional, empathetic, and constructive."
    )

    # Construct a structured prompt inserting our hard mathematical metrics
    prompt = f"""
    A customer has been flagged by our machine learning model with a {risk_score * 100:.2f}% risk of churning.

    Customer Profile Facts:
    - Age: {customer_data['Customer_Age']}
    - Total Products Held (Relationship Count): {customer_data['Total_Relationship_Count']}
    - Total Transaction Amount (Last 12 Months): ${customer_data['Total_Trans_Amt']}
    - Total Transaction Count (Last 12 Months): {customer_data['Total_Trans_Ct']}
    - Credit Card Utilization Rate: {customer_data['Avg_Utilization_Ratio'] * 100}%

    Operational Task:
    1. Briefly write an internal expert analysis explaining WHY this customer is leaving based on the facts.
    2. Design a specific retention offer tactic (e.g., cashback boost, fee waiving, or 2nd product benefits).
    3. Draft a personalized, warm, and highly converting email message to the customer written in fluent English.

    Format the output cleanly with clear headers.
    """

    # Fire the modern API call using the recommended gemini-2.5-flash model
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )

    print("\n=================== AGENT OUTPUT ===================")
    print(response.text)
    print("====================================================")


# ========================================================
# SIMULATION / TESTING ZONE
# ========================================================
if __name__ == "__main__":
    # 1. Securely fetch the API key
    secret_key = load_api_key()

    if secret_key:
        # 2. Train the analytics layer
        trained_model, trained_columns = load_and_train_model()

        # 3. Define the simulated customer profile
        high_risk_sample = {
            'Customer_Age': 44, 'Gender': 'F', 'Dependent_count': 2, 'Education_Level': 'Graduate',
            'Marital_Status': 'Single', 'Income_Category': 'Less than $40K', 'Card_Category': 'Blue',
            'Months_on_book': 36, 'Total_Relationship_Count': 1, 'Months_Inactive_12_mon': 3,
            'Contacts_Count_12_mon': 4, 'Credit_Limit': 2500, 'Total_Revolving_Bal': 0,
            'Avg_Open_To_Buy': 2500, 'Total_Amt_Chng_Q4_Q1': 0.35, 'Total_Trans_Amt': 1100,
            'Total_Trans_Ct': 22, 'Total_Ct_Chng_Q4_Q1': 0.28, 'Avg_Utilization_Ratio': 0.0
        }

        # 4. Extract the mathematical risk score
        risk_score = run_agent_inference(trained_model, trained_columns, high_risk_sample)

        # 5. Agent Gatekeeper Rule: If risk is higher than 75%, trigger the LLM agent!
        if risk_score > 0.75:
            trigger_llm_agent(high_risk_sample, risk_score, secret_key)
        else:
            print("✅ Customer risk is low. No action required.")