import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix

# A helper to render the SHAP plot in Streamlit
@st.cache_data
def get_shap_plot_as_html(base_value, shap_values, features):
    """Caches the HTML representation of a SHAP force plot."""
    # Note: Corrected the function signature
    return shap.force_plot(base_value, shap_values, features, matplotlib=False).html()

def show_feature_importance(model, X_test, model_name):
    """Displays the SHAP summary plot for a given model."""
    st.subheader(f"SHAP Feature Importance for {model_name}")
    st.markdown("This plot shows the most important features and their impact. Red dots are high feature values, blue are low.")
    
    try:
        classifier = model.named_steps['classifier']
        if 'rfe' in model.named_steps:
            support = model.named_steps['rfe'].support_
            X_test_for_shap = X_test.loc[:, support]
        else:
            X_test_for_shap = X_test
            
        if 'Lightgbm' in model_name or 'Rf' in model_name or 'Catboost' in model_name:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_for_shap)
            shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_to_plot, X_test_for_shap, plot_type="dot", show=False)
            st.pyplot(fig)
        else:
            st.warning("SHAP summary plots are currently optimized for tree-based models.")
    except Exception as e:
        st.error(f"Could not generate SHAP summary plot. Error: {e}")

def show_single_prediction_explanation(model, X_test, model_name):
    """Shows a SHAP force plot to explain a single prediction."""
    st.subheader("Explain a Single Prediction")
    instance_index = st.number_input("Select a customer index from the test set to explain:", 0, len(X_test)-1, 0)
    
    try:
        classifier = model.named_steps['classifier']
        if 'rfe' in model.named_steps:
            support = model.named_steps['rfe'].support_
            X_test_for_shap = X_test.loc[:, support]
        else:
            X_test_for_shap = X_test
            
        if 'Lightgbm' in model_name or 'Rf' in model_name or 'Catboost' in model_name:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test_for_shap)
            
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            shap_instance_values = shap_values[1][instance_index, :] if isinstance(shap_values, list) else shap_values[instance_index, :]
            
            st.write("This force plot shows how each feature pushed the prediction from the baseline to the final output.")
            st.components.v1.html(get_shap_plot_as_html(expected_value, shap_instance_values, X_test_for_shap.iloc[instance_index]), height=150)
        else:
            st.warning("Single prediction explanations are currently optimized for tree-based models.")
    except Exception as e:
        st.error(f"Could not generate SHAP force plot. Error: {e}")

# --- FUNCTION ADDED HERE ---
def show_subgroup_errors(raw_df, X_test, y_test, model, model_name):
    """Creates a stacked bar chart of error types by customer job."""
    st.subheader(f"Error Analysis by Job for {model_name}")
    
    # Use a generic try-except block to handle different model prediction methods
    try:
        if hasattr(model, 'predict_proba'):
             y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        else: # For Keras models
             y_pred = (model.predict(X_test) > 0.5).astype(int)
    except Exception as e:
        st.error(f"Could not generate predictions for subgroup analysis. Error: {e}")
        return

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
    
    fig = px.bar(outcome_by_job, x='job', y='count', color='Outcome', title="Distribution of Prediction Outcomes by Job", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)