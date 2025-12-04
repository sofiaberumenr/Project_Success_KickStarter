import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_preprocessed_data():
    """Load preprocessed training and test data."""
    print("Loading preprocessed data...")
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Train multiple classification models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} training complete!")
    
    return trained_models

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print('='*50)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Failed', 'Successful']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def save_best_model(models, results, output_dir="models"):
    """Save the best performing model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]
    
    model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"Model saved to {model_path}")
    print('='*50)

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate all models
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Save best model
    save_best_model(trained_models, results)
