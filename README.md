# Telecommunications – Call Drop Prediction

# Project Title
# Call Drop Prediction Using Machine Learning

# ----------------------------------------

# Introduction
# Call drops are a common problem in telecommunication networks.
# They affect customer satisfaction and reduce service quality.
# This project predicts whether a call will drop based on network
# and user-related factors using Machine Learning techniques.
# An interactive Streamlit dashboard is also developed for real-time prediction.

# ----------------------------------------

# Objectives
# - Predict call drop probability using Machine Learning
# - Identify the main reasons behind call drops
# - Improve network reliability using predictive analysis
# - Create a simple dashboard for real-time prediction

# ----------------------------------------

# Dataset
# A synthetic telecom dataset is used with the following features:
# - Signal Strength (dBm)
# - Network Load (%)
# - User Speed (km/h)
# - Call Duration (seconds)
# - Tower Distance (meters)
# - Packet Loss (%)
# - Dropped Call (Target Variable)
# Target variable:
# 0 -> Call Successful
# 1 -> Call Dropped

# ----------------------------------------

# Feature Engineering
# Additional features created to improve model performance:
# - Signal Quality (Poor / Moderate / Good)
# - Mobility Level (Low / Medium / High)
# - Load Category (Low / Medium / High)
# - Call Risk Score (based on network load, packet loss, and signal strength)
# Categorical features were converted into numeric form using Label Encoding

# ----------------------------------------

# Machine Learning Models

# Random Forest Classifier
# - Main model used for prediction
# - Handles complex relationships
# - Gives good accuracy

# Logistic Regression
# - Used for comparison
# - Simple and interpretable model

# ----------------------------------------

# Model Evaluation
# Metrics used to evaluate model performance:
# - Confusion Matrix
# - Accuracy
# - Precision
# - Recall
# - F1 Score

# ----------------------------------------

# Dashboard
# Streamlit dashboard features:
# - User input using sliders
# - Real-time prediction
# - Risk warnings
# - Data visualizations
# - Storage of predictions using SQLite database

# ----------------------------------------

# Technologies Used
# - Python
# - Pandas
# - NumPy
# - Scikit-learn
# - Streamlit
# - Matplotlib
# - Seaborn
# - SQLite
# - Jamovi (predictive analytics)

# ----------------------------------------

# Project Structure
# call-drop-prediction/
# ├── `call_drop.py`                # Main Streamlit app
# ├── `call_drop_prediction.py`     # ML model code
# ├── `calldrop_dataset.csv`        # Dataset
# ├── `requirements.txt`            # Dependencies
# ├── `README.md`                   # Project documentation
# └── `.gitignore`                  # Git ignore file

# ----------------------------------------

# How to Run the Project

# 1. Install required libraries
# `pip install -r requirements.txt`

# 2. Run the Streamlit app
# `streamlit run call_drop.py`

# ----------------------------------------

# Conclusion
# This project shows how Machine Learning can be used to predict call drops
# in telecom networks. By analyzing parameters such as signal strength,
# network load, and packet loss, the system identifies risky conditions
# that may lead to call failure, helping improve network reliability.
