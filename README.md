# 🎯 Customer Churn Prediction with Explainable AI (SHAP)

An interactive, end-to-end Machine Learning web application built using **Streamlit** and **XGBoost** to predict customer churn risk. The system integrates **SHAP (SHapley Additive exPlanations)** to provide transparent, feature-level interpretability for both existing and new custom profiles.

---

## ✨ Features

- **Existing Customer Mode:** Select an existing customer index from the database to instantly evaluate their real-time prediction and probability score.
- **New Customer Mode:** Input unique, custom operational metrics manually to evaluate mock risk flags dynamically.
- **Local Explanations:** Real-time **SHAP Waterfall Plots** visually map exactly which factors (e.g., Age, Products, Activity) drive individual risk up or down.
- **Global Model Insights:** On-demand **SHAP Summary/Beeswarm Plots** reveal top systemic drivers across your entire client base.

---

## 🛠️ Project Structure

```text
├── app.py                      # Main Streamlit web application
├── Customer_churn_model.pkl    # Pre-trained XGBoost classification model
├── label_encoders.pkl          # Saved categorical encoders (Geography, Gender)
├── Churn_Modelling.xls         # Customer risk database
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

Follow these instructions to set up and run the application locally on your machine.

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation & Environment Setup
Clone this repository and navigate into the project root directory:
```bash
git clone https://github.com
cd Customer_Churn_Pred_with_ExplainableAi
```

Create a clean virtual environment and activate it:
```bash
python3 -m venv ml-env
source ml-env/bin/activate
```

Install all necessary project tracking dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launching the App
Run the local Streamlit hosting server wrapper via your terminal:
```bash
streamlit run app.py
```
Your default browser will automatically open a tab displaying the live interactive engine dashboard at `http://localhost:8501`.

---

## 📊 Core Tech Stack & Libraries

- **Frontend:** [Streamlit](https://streamlit.io) (Interactive dashboard UI layout framework)
- **Model Engine:** [XGBoost](https://readthedocs.io) (High-performance gradient boosting classifier)
- **Explainability:** [SHAP](https://github.com) (Game-theoretic local and global feature attribution)
- **Data Engineering:** Pandas, Scikit-Learn, Joblib, Matplotlib
