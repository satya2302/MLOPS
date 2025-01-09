# test.py
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load the iris dataset for class names
data = load_iris()

# Load the trained model
loaded_model = joblib.load('iris_model.joblib')

if __name__ == "__main__":
    # Example usage
    example_features = [5.1, 3.5, 1.4, 0.2]
    example_features = np.array(example_features).reshape(1, -1)
    prediction = loaded_model.predict(example_features)
    print(f"The predicted {example_features.tolist()} is: {data.target_names[prediction[0]]}")
