import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load the iris dataset for class names
data = load_iris()

# Load the trained model
loaded_model = joblib.load('iris_model.joblib')

def predict_iris(features):
    """
    Predict the class of an iris flower given its features.

    Parameters:
    features (list or array): A list or array of 4 numerical features in the order:
                              [sepal length, sepal width, petal length, petal width].

    Returns:
    str: The predicted iris class.
    """
    if len(features) != 4:
        raise ValueError("Features must be a list or array of 4 numerical values.")
    
    features = np.array(features).reshape(1, -1)
    prediction = loaded_model.predict(features)
    return prediction

if __name__ == "__main__":
    # Example usage
    example_features = [5.1, 3.5, 1.4, 0.2]
    prediction = predict_iris(example_features)
    print(f"The predicted class for the given features {example_features} is: {prediction}")

