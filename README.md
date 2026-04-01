# Telco Customer Churn Analysis & Dashboard

This repository contains a comprehensive data science project focused on analyzing and predicting customer churn for a telecommunications company. It combines deep exploratory data analysis (EDA) with a high-performance machine learning pipeline and an interactive visualization dashboard.

## Project Structure

- **`Telco-Customer.ipynb`**: A complete end-to-end data science notebook.
    - Data Cleaning & Pre-processing (Handling missing values, encoding categorical variables).
    - Exploratory Data Analysis (EDA) with Seaborn and Matplotlib.
    - Model Training (Gradient Boosting Machine - GBM).
    - Model Interpretation using **LIME** (Local Interpretable Model-agnostic Explanations).
- **`telco_churn_dashboard.py`**: A production-ready **Streamlit** dashboard.
    - Interactive KPI tracking.
    - Global feature importance visualization using **SHAP** values.
    - Individual customer churn risk assessment.

---

## Key Features

### 1. Advanced Machine Learning Pipeline
The project utilizes a **Gradient Boosting Machine (GBM)** to classify customers at risk of leaving. The model is optimized to provide not just a prediction, but a probability score that can be used for tiered retention strategies.

### 2. Explainable AI (XAI)
To bridge the gap between "black-box" models and business decisions, this project implements:
- **SHAP (SHapley Additive exPlanations):** To identify which global factors (like Contract Type or Tenure) drive churn across the entire dataset.
- **LIME:** To provide granular, instance-level explanations, showing exactly why a specific customer is predicted to churn.

### 3. Interactive Dark-Theme Dashboard
The Streamlit application provides a modern UI/UX for stakeholders:
- **Real-time Metrics:** Displays Churn Rate, Average Monthly Charges, and Total Revenue.
- **Dynamic Visuals:** Built with Plotly for interactive zooming and data inspection.
- **Strategic Insights:** Highlights that "Month-to-month" contracts and "Fiber optic" internet service are the primary drivers of churn.

---

## Setup & Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### Installation
```bash
pip install pandas numpy matplotlib seaborn plotly streamlit lime shap
```

### Running the Dashboard
```bash
streamlit run telco_churn_dashboard.py
```

---

##  Business Insights Summary
* **Contract Risk:** Customers on month-to-month contracts are significantly more likely to churn compared to those on one or two-year plans.
* **Financial Impact:** Higher monthly charges correlate with increased churn, suggesting a need for price-optimization or loyalty discounts.
* **Support factor:** Lack of Online Security and Tech Support services is a strong predictor of customer departure.

---

## Author
* **Rakesh , Seongmin Choi (Edward) , MawuKo**
* AI Developer | Software Architect 

> **Note:** This project was developed as part of the AI Post-graduate program at Fanshawe College.