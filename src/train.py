# hyperparameter_tuning_gridsearch.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid for GridSearchCV
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Set up the GridSearchCV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid,
    cv=5, scoring="accuracy", n_jobs=-1
)

# Run the grid search
grid_search.fit(X_train, y_train)

# Best parameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create an input example
input_example = X_test[0].reshape(1, -1)
# Taking the first sample from X_test


# Log the final model with MLflow
mlflow.set_experiment("Random Forest Experiment")

with mlflow.start_run():
    mlflow.log_params({
        "param_grid": str(param_grid),
        # Document the grid search space
        "best_n_estimators": best_params["n_estimators"],
        # Best value for n_estimators
        "best_max_depth": best_params["max_depth"],
        # Best value for max_depth
        "best_min_samples_split": best_params["min_samples_split"],
        # Best value for min_samples_split
        "best_min_samples_leaf": best_params["min_samples_leaf"]
        # Best value for min_samples_leaf
    })
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log the model with input_example for auto-inferencing the signature
    mlflow.sklearn.log_model(best_model, "model_optimized",
                             input_example=input_example)
