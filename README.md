# 🎓 Student Performance Prediction in Online Courses

This project uses **Supervised Machine Learning** to predict whether a student will **Pass** or **Fail** an online course based on their academic and behavioral data.

---

## 📌 Problem Statement

With the rise of online learning, predicting student performance can help educators intervene early and improve academic outcomes. This project predicts the **pass/fail status** using various supervised learning algorithms and integrates the final model into a **Flask web application**.

---

## 📊 Dataset

The dataset includes features such as:
- `studytime` (hours of study)
- `failures` (number of past academic failures)
- `absences` (number of classes missed)
- `G1` (first period grade)
- `G2` (second period grade)
- Target: `pass_fail` (Pass = 1, Fail = 0)

The data is preprocessed to handle missing values, outliers, feature scaling, and encoding.

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- Flask (Backend)
- HTML, CSS, JavaScript (Frontend)
- Jupyter Notebooks

---

## 🗂 Directory Structure

Project/
├── App/
│ ├── app.py
│ ├── model/
│ │ ├── model.pkl
│ │ └── fine_tune.pkl
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ │ ├── main_css.css
│ │ └── main.js
│ └── routes/
│ └── init.py
│
├── Data/
│ ├── student_performance_2k.csv
│ └── preprocessed_data.csv
│
├── Training/
│ ├── preprocess_data.ipynb
│ └── training_notebook.ipynb
│
├── Evaluation/
│ ├── best_model_saving.ipynb
│ └── evaluation_and_tuning.ipynb
│
├── README.md
├── requirements.txt
├── python_version.txt
└── setup.exe

---

## 🔎 Machine Learning Models Used

The following algorithms were trained and evaluated:
1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **AdaBoost**
5. **Gradient Boosting Classifier**

The **best model** was selected based on accuracy and further fine-tuned using **GridSearchCV**.

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
