import streamlit as st
import pandas as pd
import plotly.express as px

def show(selected_models, models, X_test):
    """Creates the model interpretability tab."""
    st.header("Model Logic and Interpretability")
    
    if not selected_models:
        st.warning("No models selected in the sidebar.")
        return

    # --- 1. Dropdown to select a single model ---
    model_to_inspect = st.selectbox("Select a model to inspect:", options=selected_models)
    
    # --- 2. Feature Importance Plot ---
    if model_to_inspect:
        try:
            st.subheader(f"Feature Importances for {model_to_inspect}")
            classifier = models[model_to_inspect].named_steps['classifier']
            feature_names = X_test.columns
            
            if 'rfe' in models[model_to_inspect].named_steps:
                support = models[model_to_inspect].named_steps['rfe'].support_
                feature_names = X_test.columns[support]
            
            importances = classifier.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False).head(15)
            fig_imp = px.bar(importance_df.sort_values('Importance'), x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            st.warning(f"Could not automatically extract feature importances for {model_to_inspect}.")