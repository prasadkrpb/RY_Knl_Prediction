
import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rice Yield Prediction - Kurnool", layout="wide")
st.title("🌾 Rice Yield Prediction - Kurnool")

@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    vif_table = pd.read_csv("vif_table.csv")
    return rf_model, xgb_model, metrics, vif_table

rf_model, xgb_model, metrics, vif_table = load_artifacts()
features = metrics["features"]
df = pd.read_csv("Kurnool.csv")

# Model Comparison
st.subheader("📊 Model Accuracy Comparison")
metrics_df = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "R² Score": [metrics["RandomForest"]["R2"], metrics["XGBoost"]["R2"]],
    "RMSE": [metrics["RandomForest"]["RMSE"], metrics["XGBoost"]["RMSE"]]
})
st.dataframe(metrics_df.style.format({"R² Score": "{:.3f}", "RMSE": "{:.2f}"}))

# Load precomputed analysis plots
st.subheader("📈 Correlation Matrix")
st.image("correlation_matrix.png")

st.subheader("🌟 Random Forest Feature Importance")
st.image("feature_importance_rf.png")

st.subheader("🌟 XGBoost Feature Importance")
st.image("feature_importance_xgb.png")

st.subheader("📄 Variance Inflation Factor (VIF) Table")
st.dataframe(vif_table.style.format({"VIF": "{:.2f}"}))

# Prediction Section
st.subheader("🌱 Choose Model and Enter Climate Parameters for Prediction")
selected_model_name = st.selectbox("Select Model", ["Random Forest", "XGBoost"], index=1)
model = rf_model if selected_model_name == "Random Forest" else xgb_model

input_data = {}
for col in features:
    col_min, col_max = float(df[col].min()), float(df[col].max())
    default_value = round((col_min + col_max) / 2, 1)
    if col == "year": col_max += 10
    input_data[col] = st.slider(
        f"{col.replace('_', ' ').capitalize()}",
        min_value=round(col_min,1),
        max_value=round(col_max,1),
        value=default_value,
        step=0.1
    )

input_df = pd.DataFrame([input_data])
if st.button("🔍 Predict Rice Yield"):
    prediction = model.predict(input_df)
    st.success(f"🌾 Predicted Rice Yield ({selected_model_name}): **{prediction[0]:.2f} kg/ha**")
