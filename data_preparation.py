import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Connection configuration
DB_USER = 'root'
DB_HOST = '127.0.0.1'
DB_NAME = 'finguard_db'
DB_PASSWORD = 'Gekker1999'

try:
    print("Connecting to the database via SQLAlchemy...")
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(connection_string)

    query = "SELECT * FROM raw_customers;"
    df = pd.read_sql(query, engine)
    print("💥 Success! Data loaded successfully into Python.")

    # 2. Data Preprocessing
    if 'CLIENTNUM' in df.columns:
        df = df.drop(columns=['CLIENTNUM'])

    target_map = {'Existing Customer': 0, 'Attrited Customer': 1}
    df['Attrition_Flag'] = df['Attrition_Flag'].map(target_map)

    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print("Data preprocessing completed.")

    # 3. Machine Learning Modeling
    print("\nPreparing features and target variable...")
    X = df_encoded.drop(columns=['Attrition_Flag'])
    y = df_encoded['Attrition_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("💥 Model training completed successfully!")

    # 4. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Model Accuracy: {accuracy * 100:.2f}%")

    # ==========================================
    # NEW: FEATURE IMPORTANCE EXTRACTION
    # ==========================================
    print("\nExtracting feature importances...")

    # Create a DataFrame with features and their corresponding importance scores
    importances = model.feature_importances_
    feature_names = X.columns

    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort features by importance in descending order and get top 10
    top_10_features = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)

    print("\n👑 TOP 10 MOST IMPORTANT FEATURES FOR PREDICTING CHURN:")
    print("========================================================")
    for index, row in top_10_features.iterrows():
        print(f"{row['Feature']:<30} | Score: {row['Importance']:.4f} ({row['Importance'] * 100:.2f}%)")
    print("========================================================")

except Exception as e:
    print(f"\nAn error occurred: {e}")