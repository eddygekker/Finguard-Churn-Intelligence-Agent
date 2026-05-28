import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Dynamically resolve data path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'finguard_data.csv')


def train_predictive_engine():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data file not found at: {DATA_PATH}. Please run data_preparation.py first.")

    df = pd.read_csv(DATA_PATH)

    # Features & Target definition
    X = df.drop(columns=['CLIENTNUM', 'Attrition_Flag'])
    y = df['Attrition_Flag']

    # Encode categorical features
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit production model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, label_encoders, y_encoder, X_test


if __name__ == "__main__":
    train_predictive_engine()
    print("Predictive Engine module validated successfully.")