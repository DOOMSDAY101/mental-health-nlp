from flask import Flask, request, jsonify, render_template
from src.predict import predict_text

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def get_prediction():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    prediction = predict_text(text)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
