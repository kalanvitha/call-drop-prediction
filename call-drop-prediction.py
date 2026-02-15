import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# ------------------ Page Config ------------------
st.set_page_config(page_title="Call Drop Dashboard", layout="wide", page_icon="📞")
st.title("📞 Call Drop Prediction Dashboard")
st.markdown("##### Predict the likelihood of call drops based on network and user parameters.")

# ------------------ Load Dataset ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./calldrop_dataset.csv").drop_duplicates()
    df['packet_loss'] = df['packet_loss'].fillna(df['packet_loss'].mean())

    df["signal_quality"] = df["signal_strength"].apply(lambda x: "Poor" if x < -95 else "Moderate" if x < -75 else "Good")
    df["mobility_level"] = df["user_speed"].apply(lambda x: "Low" if x < 20 else "Medium" if x < 60 else "High")
    df["load_category"] = pd.cut(df["network_load"], bins=[0, 40, 70, 100], labels=["Low", "Medium", "High"])
    df["call_risk_score"] = (df["network_load"]*0.4) + (df["packet_loss"]*10) + (abs(df["signal_strength"])*0.3)

    le_signal, le_mobility, le_load = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df['signal_quality'] = le_signal.fit_transform(df['signal_quality'])
    df['mobility_level'] = le_mobility.fit_transform(df['mobility_level'])
    df['load_category'] = le_load.fit_transform(df['load_category'])

    return df, le_signal, le_mobility, le_load

df, le_signal, le_mobility, le_load = load_data()

# ------------------ Train Model ------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

X = df[['signal_strength','network_load','user_speed','call_duration',
        'tower_distance','packet_loss','signal_quality','mobility_level',
        'load_category','call_risk_score']]
y = df['dropped_call']

model = train_model(X, y)

# ------------------ Prediction History CSV ------------------
history_file = "./predictions.csv"
if os.path.exists(history_file):
    all_preds = pd.read_csv(history_file)
else:
    all_preds = pd.DataFrame(columns=["signal_strength","network_load","user_speed","call_duration",
                                      "tower_distance","packet_loss","prediction","probability",
                                      "risk_score","timestamp"])

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("🎛️ Call Parameters")
preset = st.sidebar.selectbox(
    "Choose a Scenario",
    ["Custom", "Highway Speed", "City Call", "Poor Network"]
)

if preset == "Highway Speed":
    signal_strength, network_load, user_speed, call_duration, tower_distance, packet_loss = -80, 50, 120, 300, 4000, 2
elif preset == "City Call":
    signal_strength, network_load, user_speed, call_duration, tower_distance, packet_loss = -65, 30, 30, 180, 1000, 1
elif preset == "Poor Network":
    signal_strength, network_load, user_speed, call_duration, tower_distance, packet_loss = -100, 80, 40, 600, 6000, 15
else:
    signal_strength = st.sidebar.slider("Signal Strength (dBm)", -120, 0, -80, step=1)
    network_load = st.sidebar.slider("Network Load (%)", 0, 100, 50, step=5)
    user_speed = st.sidebar.slider("User Speed (km/h)", 0, 300, 30, step=5)
    call_duration = st.sidebar.slider("Call Duration (sec)", 0, 36000, 300, step=60)
    tower_distance = st.sidebar.slider("Tower Distance (m)", 0, 10000, 5000, step=1000)
    packet_loss = st.sidebar.slider("Packet Loss (%)", 0.0, 50.0, 1.0, step=0.1)

# ------------------ Compute Features ------------------
def compute_features():
    sig_label = "Poor" if signal_strength < -95 else "Moderate" if signal_strength < -75 else "Good"
    mob_label = "Low" if user_speed < 20 else "Medium" if user_speed < 60 else "High"
    load_label = "Low" if network_load <= 40 else "Medium" if network_load <= 70 else "High"
    sig_q = le_signal.transform([sig_label])[0]
    mob = le_mobility.transform([mob_label])[0]
    load = le_load.transform([load_label])[0]
    risk_score = (network_load*0.4) + (packet_loss*10) + (abs(signal_strength)*0.3)
    return pd.DataFrame([[signal_strength, network_load, user_speed, call_duration,
                          tower_distance, packet_loss, sig_q, mob, load, risk_score]], columns=X.columns), risk_score

input_df, risk_score = compute_features()

# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🔮 Prediction", "🗄️ History"])

# ---- Analysis Tab ----
with tab1:
    st.subheader("Signal Strength Distribution")
    fig, ax = plt.subplots(figsize=(8,3))
    sns.histplot(df['signal_strength'], bins=20, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("Signal Strength (dBm)")
    ax.set_ylabel("Number of Calls")
    st.pyplot(fig)

# ---- Prediction Tab ----
with tab2:
    if st.button("Predict Call Drop"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Append to history CSV
        new_pred = {
            "signal_strength": signal_strength,
            "network_load": network_load,
            "user_speed": user_speed,
            "call_duration": call_duration,
            "tower_distance": tower_distance,
            "packet_loss": packet_loss,
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_score": float(risk_score),
            "timestamp": datetime.now()
        }
        all_preds = pd.concat([all_preds, pd.DataFrame([new_pred])], ignore_index=True)
        all_preds.to_csv(history_file, index=False)

        st.subheader("Prediction Result")
        st.success("✅ Call is Likely Successful" if prediction == 0 else "⚠️ Call is Likely to Drop")

# ---- History Tab ----
with tab3:
    st.subheader("📋 All Predictions History")
    st.dataframe(all_preds)
    st.download_button("⬇️ Download CSV", all_preds.to_csv(index=False), "predictions.csv")
