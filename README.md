# Telecommunications – Call Drop Prediction

## Project Title
Call Drop Prediction Using Machine Learning

---

## Introduction

Call drops are a common problem in telecommunication networks. They affect customer satisfaction and reduce service quality.  

This project focuses on predicting whether a call will drop based on different network and user-related factors using Machine Learning techniques.

An interactive dashboard is also developed to make predictions in real time.

---

## Objectives

- To predict call drop probability using Machine Learning
- To identify the main reasons behind call drops
- To improve network reliability using predictive analysis
- To create a simple dashboard for real-time prediction

---

## Dataset

A synthetic telecom dataset is used for this project.  
The dataset includes the following features:

- Signal Strength (dBm)
- Network Load (%)
- User Speed (km/h)
- Call Duration (seconds)
- Tower Distance (meters)
- Packet Loss (%)
- Dropped Call (Target Variable)

Target variable:
- 0 → Call Successful  
- 1 → Call Dropped  

---

## Feature Engineering

To improve model performance, additional features were created:

- Signal Quality (Poor / Moderate / Good)
- Mobility Level (Low / Medium / High)
- Load Category (Low / Medium / High)
- Call Risk Score based on network load, packet loss and signal strength

Categorical features were converted into numeric form using Label Encoding.

---

## Machine Learning Models

Two models were implemented:

### Random Forest Classifier
- Main model used for prediction
- Handles complex relationships
- Gives good accuracy

### Logistic Regression
- Used for comparison
- Simple and interpretable model

---

## Model Evaluation

The performance of the models was evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score

These metrics help measure how well the model predicts call drops.

---

## Dashboard

A Streamlit dashboard was developed with the following features:

- User input using sliders
- Real-time prediction
- Risk warnings
- Data visualizations
- Storage of predictions using SQLite database

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- SQLite

---

## Project Structure

call-drop-prediction/

- call_drop.py  
- call_drop_prediction.py  
- calldrop_dataset.csv  
- requirements.txt  
- README.md  
- .gitignore  

---

## How to Run the Project

1. Install required libraries:
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run call_drop.py

---

## Conclusion

This project shows how Machine Learning can be used to predict call drops in telecom networks.  

By analyzing network parameters such as signal strength, load, and packet loss, the system helps in identifying risky conditions that may lead to call failure.
