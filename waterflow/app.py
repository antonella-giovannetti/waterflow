from flask import Flask, request, render_template
import mlflow.pyfunc
import pandas as pd
import mlflow
import mlflow.xgboost


app = Flask(__name__)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "water_potability_xgb"
MODEL_VERSION = 1

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.xgboost.load_model(model_uri)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    if request.method == "POST":
        try:
            input_data = {
                key: float(request.form[key]) for key in request.form
            }
            input_df = pd.DataFrame([input_data])
            proba = model.predict_proba(input_df)[0]
            prediction = int(proba[1] >= 0.5) 
            probability = round(proba[1 if prediction == 1 else 0] * 100, 2)  
        except Exception as e:
            prediction = f"Erreur: {str(e)}"
            probability = None

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
