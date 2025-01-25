from flask import Flask, request, jsonify
import mlflow
import joblib
import mlflow.sklearn
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model from MLflow
# Replace <RUN_ID> with the actual MLflow run ID for the best model
#model = mlflow.sklearn.load_model("mlruns/572425099079902757/5bc902453bba4db8ad936ab04a6b3119/artifacts/model_optimized")
model = joblib.load('iris_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the features from the POST request
        data = request.json
        features = np.array(data['features']).reshape(1, -1)  # Reshape to match input format
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return the prediction as a JSON response
        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True,host='0.0.0.0')
