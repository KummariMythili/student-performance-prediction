from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/fine_tune.pkl')

# Manually define mean and std for each feature (replace with your actual values)
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
        # Get inputs from form
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])
        absences = float(request.form['absences'])
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])

        # Derived features
        avg_grade = (G1 + G2) / 2
        studytime_level = 0 if studytime < 2 else 1

        # Manual scaling (equivalent to StandardScaler)
        def scale(val, mean, std):
            return (val - mean) / std

        features_scaled = [
            scale(studytime, feature_means['studytime'], feature_stds['studytime']),
            scale(failures, feature_means['failures'], feature_stds['failures']),
            scale(absences, feature_means['absences'], feature_stds['absences']),
            scale(G1, feature_means['G1'], feature_stds['G1']),
            scale(G2, feature_means['G2'], feature_stds['G2']),
            scale(avg_grade, feature_means['avg_grade'], feature_stds['avg_grade']),
            studytime_level  # no scaling for binary feature
        ]

        prediction = model.predict([features_scaled])[0]
        result = "✅ Pass" if prediction == 1 else "❌ Fail"

        return render_template('index.html',
                               prediction_text=f"The student is likely to: {result}",
                               studytime=studytime,
                               failures=failures,
                               absences=absences,
                               G1=G1,
                               G2=G2)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
