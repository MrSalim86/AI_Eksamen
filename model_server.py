from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 Tilføj dette
import joblib
import pandas as pd

# Indlæs alle modeller
pipeline = joblib.load("Exam_pipeline.pkl")
model = joblib.load("Exam_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
numerisk_pipeline = pipeline.named_transformers_["num"]

# Start app
app = Flask(__name__)
CORS(app)  # 👈 Tillader alle origins (CORS)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Data fra klient
        data = request.get_json()

        # Konverter til DataFrame
        df = pd.DataFrame([data])

        # Beregn cluster
        num_data = df[["age", "avg_glucose_level", "bmi"]]
        num_transformed = numerisk_pipeline.transform(num_data)
        df["KMeans_cluster"] = str(kmeans.predict(num_transformed)[0])

        # Transformér hele input
        X_transformed = pipeline.transform(df)

        # Forudsig
        prediction = int(model.predict(X_transformed)[0])

        return jsonify({"stroke_risk": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
