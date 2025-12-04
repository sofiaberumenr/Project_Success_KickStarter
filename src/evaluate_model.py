import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path="models/best_model.joblib"):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model

def predict(model, X):
    """Make predictions using the trained model."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Failed', 'Successful'],
                yticklabels=['Failed', 'Successful'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    # Load model and test data
    model = load_model()
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    # Make predictions
    predictions, probabilities = predict(model, X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                                target_names=['Failed', 'Successful']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, predictions)
