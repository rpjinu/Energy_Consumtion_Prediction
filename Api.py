from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "model.pkl"  # Replace with your model's path
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please ensure 'model.pkl' exists.")

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Appliance Energy Consumption Prediction API!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Expect JSON input
    data = request.get_json()

    # Check if all required features are provided
    required_features = ["temperature", "humidity", "appliance_type", "time_of_day"]  # Replace with your actual feature names
    if not all(feature in data for feature in required_features):
        return jsonify({"error": "Missing required features.", "required": required_features}), 400

    try:
        # Prepare input data for the model
        input_data = np.array([
            data["temperature"],
            data["humidity"],
            data["appliance_type"],
            data["time_of_day"]
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_energy = prediction[0]  # Assuming the model outputs a single value

        return jsonify({"predicted_energy_consumption": predicted_energy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
