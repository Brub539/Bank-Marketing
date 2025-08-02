import streamlit as st

def show():
    """Creates the introduction page."""
    st.title("Bank Marketing Predictive Model Analysis")
    st.markdown("""
        This dashboard presents a comprehensive analysis of various machine learning models
        built to predict the success of a bank's telemarketing campaign.
        
        **Use the sidebar on the left to select which models you'd like to compare.**
        
        The analysis is divided into several tabs:
        - **Performance Overview:** Compares key metrics like F1-Score, Precision, and Recall.
        - **Interpretability:** Dives into the logic of individual models to understand *why* they make their predictions.
    """)