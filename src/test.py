import numpy as np
import joblib
from sklearn.datasets import load_iris
import warnings


# Suppress version mismatch warnings
warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator",
    category=UserWarning,
)

# Load the iris dataset for class names
data = load_iris()

# Load the trained model
loaded_model = joblib.load('iris_model.joblib')

if __name__ == "__main__":
    # Example usage
    example_features = [5.1, 3.5, 1.4, 0.2]
    example_features = np.array(example_features).reshape(1, -1)
    prediction = loaded_model.predict(example_features)
    predicted_class = data.target_names[prediction[0]]

    # Validate prediction
    assert predicted_class in ["setosa", "versicolor", "virginica"], \
        f"Unexpected prediction: {predicted_class}"

    print(f"The predicted is: {predicted_class}")
