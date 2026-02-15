telecommunications – call drop prediction

project title: call drop prediction using machine learning

introduction
call drops are a common problem in telecommunication networks
they affect customer satisfaction and reduce service quality
this project predicts whether a call will drop based on network
and user-related factors using machine learning techniques
an interactive streamlit dashboard is also developed for real-time prediction

objectives

predict call drop probability using machine learning

identify the main reasons behind call drops

improve network reliability using predictive analysis

create a simple dashboard for real-time prediction

dataset
a synthetic telecom dataset is used with the following features

signal strength (dbm)

network load (%)

user speed (km/h)

call duration (seconds)

tower distance (meters)

packet loss (%)

dropped call (target variable)

target variable

0 call successful

1 call dropped

feature engineering
additional features created to improve model performance

signal quality poor moderate good

mobility level low medium high

load category low medium high

call risk score based on network load packet loss and signal strength

categorical features were converted into numeric form using label encoding

machine learning models

random forest classifier

main model used for prediction

handles complex relationships

gives good accuracy

logistic regression

used for comparison

simple and interpretable model

model evaluation
metrics used to evaluate model performance

confusion matrix

accuracy

precision

recall

f1 score

dashboard
streamlit dashboard features

user input using sliders

real-time prediction

risk warnings

data visualizations

storage of predictions using sqlite database

technologies used

python

pandas

numpy

scikit-learn

streamlit

matplotlib

seaborn

sqlite

jamovi predictive analytics

project structure
call-drop-prediction

call_drop.py main streamlit app

call_drop_prediction.py ml model code

calldrop_dataset.csv dataset

requirements.txt dependencies

readme.md project documentation

.gitignore git ignore file

how to run the project
1 install required libraries
pip install -r requirements.txt

2 run the streamlit app
streamlit run call_drop.py

conclusion
this project shows how machine learning can be used to predict call drops
in telecom networks by analyzing parameters such as signal strength
network load and packet loss the system identifies risky conditions
that may lead to call failure helping improve network reliability
