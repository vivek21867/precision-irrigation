"""Streamlit entry point: registers the app's pages.

Run from the project root:
    streamlit run app/main.py
"""
import streamlit as st

pages = [
    st.Page("dashboard.py", title="Dashboard", icon="💧", default=True),
    st.Page("pages/1_Dataset_Explorer.py", title="Dataset Explorer", icon="📊"),
    st.Page("pages/2_Model_Training.py", title="Model Training", icon="🧠"),
    st.Page("pages/3_Prediction.py", title="Prediction", icon="🔮"),
    st.Page("pages/4_Model_Comparison.py", title="Model Comparison", icon="📈"),
    st.Page("pages/5_Explainability.py", title="Explainability", icon="🔍"),
    st.Page("pages/6_Settings.py", title="Settings", icon="⚙️"),
]
st.navigation(pages).run()
