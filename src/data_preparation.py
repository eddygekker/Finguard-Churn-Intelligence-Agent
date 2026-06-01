import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def train_and_save_model(csv_path="data/BankChurners.csv"):
    print("[TRAINING] Loading and preparing data for training...")
    df = pd.read_csv(csv_path)

    columns_to_drop = [
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'CLIENTNUM'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Encoding target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df['Attrition_Flag'])
    X = df.drop(columns=['Attrition_Flag'])

    # Encoding categorical features and saving encoders
    categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[TRAINING] Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Creating models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Saving artifacts as .pkl files
    print("[TRAINING] Saving artifacts to 'models/' directory...")
    with open("models/random_forest_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("models/label_encoders.pkl", "wb") as encoder_file:
        pickle.dump(encoders, encoder_file)

    print("[TRAINING] Success! Model and Encoders saved successfully.")


if __name__ == "__main__":
    train_and_save_model()