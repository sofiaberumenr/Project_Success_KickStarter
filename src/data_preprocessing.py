import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Load the Kickstarter dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    return df

def clean_data(df):
    """Clean and prepare the dataset."""
    print("Cleaning data...")
    
    # Filter only successful and failed projects
    df = df[df['state'].isin(['successful', 'failed'])].copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna(subset=['state'])
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df

def feature_engineering(df):
    """Create and select features for the model."""
    print("Engineering features...")
    
    # Example features - adjust based on your actual dataset columns
    features = []
    
    # Numeric features
    numeric_cols = ['goal', 'pledged', 'backers_count', 'usd_pledged']
    for col in numeric_cols:
        if col in df.columns:
            features.append(col)
    
    # Categorical features to encode
    categorical_cols = ['category', 'country', 'currency']
    for col in categorical_cols:
        if col in df.columns:
            features.append(col)
    
    return features

def preprocess_data(df, features):
    """Preprocess features and target variable."""
    print("Preprocessing data...")
    
    # Prepare feature matrix
    X = df[features].copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Prepare target variable
    y = (df['state'] == 'successful').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Success rate in training: {y_train.mean():.2%}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders

if __name__ == "__main__":
    # Load and preprocess data
    data_path = "data/kickstarter_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please place your Kickstarter dataset in the data/ directory")
    else:
        df = load_data(data_path)
        df = clean_data(df)
        features = feature_engineering(df)
        
        X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(df, features)
        
        # Save preprocessed data
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/X_train.npy", X_train)
        np.save("data/processed/X_test.npy", X_test)
        np.save("data/processed/y_train.npy", y_train)
        np.save("data/processed/y_test.npy", y_test)
        
        print("\nPreprocessing complete! Data saved to data/processed/")
