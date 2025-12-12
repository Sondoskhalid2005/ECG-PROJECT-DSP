from unittest import result

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from preprocessing import input_signal_preprocessing
# import  from model file the final  

st.title("ECG VS PVC classifier")
with st.form(key="ECG_FORM"):
    st.subheader("Upload ECG signal")
    signal=st.file_uploader("Upload the First Signal", type=["txt"])
    submit=st.form_submit_button(label="Classify")
    if submit:
        if not signal :
            st.warning("Please Fill all Fields!")
        else:
            signal=input_signal_preprocessing(signal)
            st.subheader("ECG signal plot")
            fig = go.Figure()
            indx = np.arange(len(signal), dtype=int)
            fig.add_trace(go.Scatter(x=indx,y=signal,mode='lines+markers'))
            fig.update_layout(xaxis_title='Index',yaxis_title='Samples',)
            st.plotly_chart(fig)
            #result=model(signal)
            st.write(f"ECG input signal classification is : ${result}")
            
            