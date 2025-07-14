### ✅ Complete Flask App with Debugging and Input Confirmation
# File: App/app.py

from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ✅ Load trained model
model = joblib.load(os.path.join("model", "fine_tune.pkl"))

# ❗ Optional: Load scaler if used during training (uncomment if needed)
# scaler = joblib.load(os.path.join("model", "scaler.pkl"))

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Collect inputs from form
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])
        absences = float(request.form['absences'])
        g1 = float(request.form['G1'])
        g2 = float(request.form['G2'])

        # ✅ Prepare input for model
        input_data = np.array([[studytime, failures, absences, g1, g2]])
        print("🔍 Input from form:", input_data)

        # ❗ If you used a scaler during training, apply it here
        # input_data = scaler.transform(input_data)

        # ✅ Make prediction
        raw_prediction = model.predict(input_data)[0]
        print("✅ Raw Prediction from model:", raw_prediction)

        # ✅ Map prediction
        prediction_label = "Pass" if raw_prediction == 1 else "Fail"

        return render_template("index.html",
                               prediction=prediction_label,
                               studytime=studytime,
                               failures=failures,
                               absences=absences,
                               g1=g1,
                               g2=g2)
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return render_template("index.html", prediction=f"⚠️ Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)