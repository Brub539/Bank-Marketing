import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# (The other functions in this file, get_shap_plot_as_html, etc., remain the same)
# ...
# ... (Previous functions go here) ...
@st.cache_data
def get_shap_plot_as_html(base_value, shap_values, features):
    """Caches the HTML representation of a SHAP force plot."""
    return shap.force_plot(base_value, shap_values, features, matplotlib=False).html()

def show_feature_importance(model, X_test, model_name):
    # ... (This function remains unchanged)
    pass

def show_single_prediction_explanation(model, X_test, model_name):
    # ... (This function remains unchanged)
    pass

def show_subgroup_errors(raw_df, X_test, y_test, model, model_name):
    """Creates a stacked bar chart of error types by customer job with user controls."""
    st.subheader(f"Error Analysis by Job for {model_name}")

    # --- KEY CHANGE: Use Session State to remember the chart mode ---
    # 1. Initialize the state if it doesn't exist (defaults to 'Count')
    if 'error_chart_mode' not in st.session_state:
        st.session_state.error_chart_mode = 'Count'

    # 2. Create the radio button. Its state is now controlled by session_state.
    # The `key` argument is crucial. It links the widget to the session state variable.
    # Streamlit handles the update automatically when the user clicks.
    chart_mode = st.radio(
        "Chart Mode:",
        ('Count', 'Percentage'),
        key='error_chart_mode', # This key links the widget to session state
        horizontal=True # A cleaner horizontal layout for the buttons
    )
    # --- END OF KEY CHANGE ---
    
    # Prediction logic (remains the same)
    is_neural_net = 'Tabnet' in model_name or isinstance(model, tf.keras.Model)
    X_test_input = X_test.to_numpy() if is_neural_net else X_test
    try:
        if isinstance(model, tf.keras.Model):
            y_pred = (model.predict(X_test_input) > 0.5).astype(int).ravel()
        else:
            y_pred = model.predict(X_test_input)
    except Exception as e:
        st.error(f"Could not generate predictions. Error: {e}")
        return

    # Data preparation (remains the same)
    test_jobs = raw_df.iloc[X_test.index]['job']
    error_df = pd.DataFrame({'job': test_jobs, 'y_true': y_test, 'y_pred': y_pred})
    conditions = [
        (error_df['y_true'] == 1) & (error_df['y_pred'] == 1),
        (error_df['y_true'] == 0) & (error_df['y_pred'] == 1),
        (error_df['y_true'] == 1) & (error_df['y_pred'] == 0),
    ]
    choices = ['Correctly Identified', 'Wrongly Targeted', 'Missed Opportunity']
    error_df['Outcome'] = np.select(conditions, choices, default='Correctly Ignored')
    outcome_by_job = error_df[error_df['Outcome'] != 'Correctly Ignored'].groupby(['job', 'Outcome']).size().reset_index(name='count')

    # Plotting logic (now simplified as the state is handled automatically)
    if st.session_state.error_chart_mode == 'Percentage':
        fig = px.histogram(
            outcome_by_job, x='job', y='count', color='Outcome',
            histfunc='sum', barnorm='percent',
            title="Proportional Distribution of Prediction Outcomes by Job"
        )
    else: # 'Count'
        fig = px.bar(
            outcome_by_job, x='job', y='count', color='Outcome',
            title="Count of Prediction Outcomes by Job", barmode='stack'
        )
    
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)