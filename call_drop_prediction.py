import pandas as pd
df=pd.read_csv("calldrop_dataset.csv")
print(df)
df.head()
df.info()
df.describe()
df.isnull().sum()
df.drop_duplicates(inplace=True)

df['packet_loss'] = df['packet_loss'].fillna(df['packet_loss'].mean())
print(df)

def signal_quality(signal):
    if signal < -95:
        return "Poor"
    elif signal < -75:
        return "Moderate"
    else:
        return "Good"

df["signal_quality"] = df["signal_strength"].apply(signal_quality)

def mobility_level(speed):
    if speed < 20:
        return "Low"
    elif speed < 60:
        return "Medium"
    else:
        return "High"

df["mobility_level"] = df["user_speed"].apply(mobility_level)

df["load_category"] = pd.cut(
    df["network_load"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"]
)

df["load_category"] = pd.cut(
    df["network_load"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"]
)
df["call_risk_score"] = (
    (df["network_load"] * 0.4) +
    (df["packet_loss"] * 10) +
    (abs(df["signal_strength"]) * 0.3)
)
df.head()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['signal_quality']=label_encoder.fit_transform(df['signal_quality'])
df['mobility_level']=label_encoder.fit_transform(df['mobility_level'])
df['load_category']=label_encoder.fit_transform(df['load_category'])
print(df)

X = df[
    [
        'signal_strength',
        'network_load',
        'user_speed',
        'call_duration',
        'tower_distance',
        'packet_loss',
        'signal_quality',
        'mobility_level',
        'load_category',
        'call_risk_score'
    ]
]

y = df['dropped_call']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,f1_score, recall_score, classification_report, roc_curve, auc

# Use your existing X_train, X_test, y_train, y_test from your Random Forest code
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Predictions
y_pred_log = log_model.predict(X_test)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]

cm_log = confusion_matrix(y_test, y_pred_log)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)
print("Confusion Matrix:")
print(cm_log)

# Metrics
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log)
recall_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print(f"\nAccuracy:  {accuracy_log:.4f}")
print(f"Precision: {precision_log:.4f}")
print(f"Recall:    {recall_log:.4f}")
print(f"F1-Score:  {f1_log:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

import matplotlib.pyplot as plt

# 1. Histogram - Signal Strength
plt.hist(df['signal_strength'], bins=20, color='skyblue', edgecolor='black', label='Signal Strength')
plt.title("Distribution of Signal Strength")
plt.xlabel("Signal Strength")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# 2. Scatter Plot - Signal Strength vs Call Duration
plt.scatter(df['signal_strength'], df['call_duration'], color='red', alpha=0.5, label='Calls')
plt.title("Signal Strength vs Call Duration")
plt.xlabel("Signal Strength")
plt.ylabel("Call Duration")
plt.legend()
plt.show()
# 3. Bar Plot - Average Call Risk Score by Load Category
avg_risk = df.groupby('load_category')['call_risk_score'].mean()
plt.bar(avg_risk.index, avg_risk.values, color='green', label='Average Risk Score')
plt.title("Average Call Risk Score by Load Category")
plt.xlabel("Load Category")
plt.ylabel("Average Call Risk Score")
plt.legend()
plt.show()


# Separate dropped and not-dropped calls
dropped = df[df['dropped_call']==1]
not_dropped = df[df['dropped_call']==0]

plt.scatter(not_dropped['signal_strength'], not_dropped['call_duration'], color='green', alpha=0.5, label='Not Dropped')
plt.scatter(dropped['signal_strength'], dropped['call_duration'], color='red', alpha=0.5, label='Dropped')

plt.title("Signal Strength vs Call Duration (Dropped Calls Highlighted)")
plt.xlabel("Signal Strength")
plt.ylabel("Call Duration")
plt.legend()
plt.show()
