import streamlit as st
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def show_threshold_slider(y_probas, y_true, model_name):
    """
    Creates an interactive slider to explore the effect of the decision threshold.
    """
    st.subheader(f"Decision Threshold Simulation for '{model_name}'")
    st.markdown("Adjust the slider to see how changing the probability threshold for a 'Yes' prediction impacts key business metrics.")

    # --- ADDITION: The interactive slider (Tool #7) ---
    decision_threshold = st.slider(
        "Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5, # Default value
        step=0.01
    )

    # Calculate predictions based on the slider's value
    y_pred_tuned = (y_probas >= decision_threshold).astype(int)

    # Calculate metrics for this threshold
    p = precision_score(y_true, y_pred_tuned, zero_division=0)
    r = recall_score(y_true, y_pred_tuned, zero_division=0)
    f1 = f1_score(y_true, y_pred_tuned, zero_division=0)

    # Display the results in KPI cards
    st.markdown("---")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Precision", f"{p:.2f}")
    kpi_cols[1].metric("Recall", f"{r:.2f}")
    kpi_cols[2].metric("F1-Score", f"{f1:.2f}")