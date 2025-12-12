import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import joblib

from preprocessing import input_signal_preprocessing
from features import build_feature_vector

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("ECG vs PVC Classifier")

with st.form(key="ECG_FORM"):
    st.subheader("Upload ECG signal (txt)")
    signal = st.file_uploader("Choose signal", type=["txt"])
    submit = st.form_submit_button(label="Classify")

    if submit:
        if not signal:
            st.warning("Please upload a file!")
        else:
            # 1) Preprocessing signal
            beat = input_signal_preprocessing(signal)

            # 2) Extract features
            feat = build_feature_vector(beat)

            # 3) Scale features
            feat_scaled = scaler.transform([feat])

            # 4) Prediction
            pred = model.predict(feat_scaled)[0]
            label = "Normal Beat" if pred == 0 else "PVC Beat"

            # 5) Plot Signal
            st.subheader("ECG Signal Plot")
            fig = go.Figure()
            idx = np.arange(len(beat))
            fig.add_trace(go.Scatter(x=idx, y=beat, mode="lines"))
            fig.update_layout(xaxis_title="Index", yaxis_title="Value")
            st.plotly_chart(fig)

            # 6) Output
            st.subheader("Classification Result:")
            st.success(label)
