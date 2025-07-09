from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/fine_tune.pkl')

# Manually define mean and std for each feature (from preprocessing step)
# ➔ You need to replace these values with the actual mean and std from your training data
feature_means = {
    'studytime': 2.5,
    'failures': 0.3,
    'absences': 4.5,
    'G1': 10.5,
    'G2': 11.0,
    'avg_grade': 10.75
}

feature_stds = {
    'studytime': 1.2,
    'failures': 0.8,
    'absences': 5.2,
    'G1': 3.5,
    'G2': 3.6,
    'avg_grade': 3.55
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])
        absences = float(request.form['absences'])
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])

        # Feature engineering
        avg_grade = (G1 + G2) / 2
        studytime_level = 0 if studytime < 2 else 1

        # Manual scaling (same as StandardScaler)
        def scale(value, mean, std):
            return (value - mean) / std

        studytime_scaled = scale(studytime, feature_means['studytime'], feature_stds['studytime'])
        failures_scaled = scale(failures, feature_means['failures'], feature_stds['failures'])
        absences_scaled = scale(absences, feature_means['absences'], feature_stds['absences'])
        G1_scaled = scale(G1, feature_means['G1'], feature_stds['G1'])
        G2_scaled = scale(G2, feature_means['G2'], feature_stds['G2'])
        avg_grade_scaled = scale(avg_grade, feature_means['avg_grade'], feature_stds['avg_grade'])

        # Final feature array (⚠ Order must match training!)
        features = np.array([[studytime_scaled, failures_scaled, absences_scaled, G1_scaled, G2_scaled, avg_grade_scaled, studytime_level]])

        # Predict
        prediction = model.predict(features)[0]
        result = "Pass" if prediction == 1 else "Fail"

        return render_template('index.html', prediction_text=f"The student is likely to: {result}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
