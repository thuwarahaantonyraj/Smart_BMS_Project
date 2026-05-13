
## cd "C:\Users\asus\Desktop\Smart_BMS_Project"
## dir
##streamlit run bms_app.py

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load ML models and scaler
soh_model = joblib.load("soh_model.pkl")
life_model = joblib.load("life_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Smart BMS Dashboard", layout="wide")

st.title("🔋 Smart BMS – Battery Health Monitoring Dashboard")
st.write("Real-time State of Health (SOH) & remaining life predictions using Machine Learning.")

# ===============================
# Input Section
# ===============================
st.subheader("📥 Sensor Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    voltage = st.slider("Voltage (V)", 2.8, 4.2, 3.9)
    soc = st.slider("State of Charge (%)", 0, 100, 80)

with col2:
    current = st.slider("Current (A)", 0.0, 5.0, 1.0)
    cycle_count = st.slider("Cycle Count", 0, 1000, 150)

with col3:
    temperature = st.slider("Temperature (°C)", 10, 60, 30)
    internal_resistance = st.slider("Internal Resistance (Ω)", 0.01, 0.20, 0.05)

# ===============================
# Prediction
# ===============================
sample = np.array([[voltage, current, temperature, soc, cycle_count, internal_resistance]])
sample_scaled = scaler.transform(sample)

soh = soh_model.predict(sample_scaled)[0]
life = life_model.predict(sample_scaled)[0]


# ===============================
# Color Logic for SOH & Life
# ===============================
def get_color(value):
    if value > 80:
        return "🟢"
    elif value > 50:
        return "🟡"
    else:
        return "🔴"

soh_color = get_color(soh)
life_color = get_color(life)


# ===============================
# Output Metrics
# ===============================
colA, colB = st.columns(2)

with colA:
    st.metric("🔋 State of Health (SOH)", f"{soh:.2f}%", delta=None)
    st.write(f"Status: {soh_color} **{ 'Healthy' if soh_color=='🟢' else 'Moderate' if soh_color=='🟡' else 'Critical' }**")

with colB:
    st.metric("⏳ Remaining Life (%)", f"{life:.2f}%", delta=None)
    st.write(f"Status: {life_color} **{ 'Long Life' if life_color=='🟢' else 'Mid Life' if life_color=='🟡' else 'End of Life' }**")


# ===============================
# Alerts (Safety System)
# ===============================
st.subheader("⚠️ Safety Alerts")

alerts = []

if temperature > 45:
    alerts.append("🔥 **Overheating detected! (>45°C)**")

if current > 3.5:
    alerts.append("⚡ **High current flow! (>3.5A)**")

if internal_resistance > 0.15:
    alerts.append("🛑 **Internal resistance is abnormally high!**")

if soh < 50:
    alerts.append("🔋 **SOH is critically low. Consider battery replacement.**")

if len(alerts) == 0:
    st.success("All conditions normal. Battery operating safely.")
else:
    for a in alerts:
        st.error(a)


# ===============================
# Graph: SOH vs Cycle Count (Prediction Curve)
# ===============================
st.subheader("📉 Predicted SOH Degradation Curve")

cycles = np.arange(0, 1000)
curve_input = np.array([[voltage, current, temperature, soc, c, internal_resistance] for c in cycles])
curve_scaled = scaler.transform(curve_input)
soh_curve = soh_model.predict(curve_scaled)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(cycles, soh_curve)
ax.set_title("SOH vs Cycle Count")
ax.set_xlabel("Cycle Count")
ax.set_ylabel("Predicted SOH (%)")
st.pyplot(fig)

st.write("---")
st.caption("Smart BMS powered by Machine Learning")
