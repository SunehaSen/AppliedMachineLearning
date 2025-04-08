from flask import Flask, request, jsonify
import joblib
from score import score
app = Flask(__name__)

# Load the trained model
model = joblib.load("/content/best_model.pkl")

@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    text = data["text"]
    threshold = data.get("threshold", 0.5)  # Default threshold is 0.5

    prediction, propensity = score(text, model, threshold)

    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })

if __name__ == "__main__":
    app.run(debug=True)