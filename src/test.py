import pytest
import numpy as np
import joblib
from sklearn.datasets import load_iris


# Load the iris dataset for class names
data = load_iris()

# Load the trained model
loaded_model = joblib.load('iris_model.joblib')


@pytest.fixture
def example_features():
    """Fixture to provide example features for prediction."""
    return np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)


def test_model_prediction(example_features):
    """Test model prediction for example features."""
    prediction = loaded_model.predict(example_features)
    predicted_class = data.target_names[prediction[0]]
    
    # Validate prediction
    assert predicted_class in ["setosa", "versicolor", "virginica"], \
        f"Unexpected prediction: {predicted_class}"


def test_model_loading():
    """Test if the model loads correctly."""
    assert loaded_model is not None, "Model failed to load"
