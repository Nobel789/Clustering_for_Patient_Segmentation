import streamlit as st

st.set_page_config(
    page_title="Healthcare Clustering Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Healthcare Clustering Studio")
st.markdown(
    """
Welcome! Use the sidebar to switch between clustering workflows:

- **🧍 Patient Segmentation** with StandardScaler + Elbow + K-Means + PCA visualization.
- **💉 Vaccination Planning** with Hierarchical Clustering and PCA + K-Means comparison.

Upload your own CSV data on each page and explore insights interactively.
"""
)

st.info("Select a page from the sidebar to start.")
