# Machine Learning Project Template

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model

# Step 1: Load Data
def load_data(filepath):
    """
    Load dataset from a CSV file.
    :param filepath: Path to the CSV file
    :return: Pandas DataFrame
    """
    data = pd.read_csv(filepath)
    return data

# Step 2: Preprocess Data
def preprocess_data(data, target_column):
    """
    Split data into features and target, then apply preprocessing.
    :param data: DataFrame containing the dataset
    :param target_column: Name of the target column
    :return: Processed feature matrix, target vector, and scaler
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# Step 3: Train Model
def train_model(X_train, y_train):
    """
    Train a machine learning model.
    :param X_train: Feature matrix for training
    :param y_train: Target vector for training
    :return: Trained model
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    :param model: Trained machine learning model
    :param X_test: Feature matrix for testing
    :param y_test: Target vector for testing
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Save Model
def save_model(model, scaler, model_filepath, scaler_filepath):
    """
    Save the trained model and scaler to disk.
    :param model: Trained machine learning model
    :param scaler: Preprocessing scaler
    :param model_filepath: Path to save the model
    :param scaler_filepath: Path to save the scaler
    """
    joblib.dump(model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    print(f"Model saved to {model_filepath}")
    print(f"Scaler saved to {scaler_filepath}")

# Main Function
if __name__ == "__main__":
    # Filepath to dataset
    filepath = "data.csv"
    target_column = "target"
    
    # Steps
    data = load_data(filepath)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_column)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, scaler, "model.pkl", "scaler.pkl")
