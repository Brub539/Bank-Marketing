import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Import our custom modules
from dashboard.data_loader import load_all
from dashboard.plotting import kpi_cards, performance_charts, strategic_charts, interpretability_charts

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Bank Marketing Model Dashboard", page_icon="📊")

# --- 2. Data Loading & Initial Prediction Generation ---
# This block runs only once at the beginning of a user's session.
if 'app_loaded' not in st.session_state:
    with st.spinner("Loading data and generating initial predictions for all models..."):
        # Load all data and models
        X_test, y_test, raw_df, models = load_all()
        
        # Calculate performance for ALL models
        summary_data = []
        predictions_temp = {}
        for model_name, model in models.items():
            is_neural_net = 'Tabnet' in model_name or isinstance(model, tf.keras.Model)
            X_test_input = X_test.to_numpy() if is_neural_net else X_test
            
            if isinstance(model, tf.keras.Model):
                p_yes = model.predict(X_test_input).ravel()
                probas = np.vstack((1 - p_yes, p_yes)).T
            else:
                probas = model.predict_proba(X_test_input)
            predictions_temp[model_name] = {'probas': probas}

            report = classification_report(y_test, (probas[:, 1] > 0.5).astype(int), output_dict=True, zero_division=0)
            summary_data.append({'Model': model_name, 'F1-Score (Yes)': report['1']['f1-score'], 'Precision (Yes)': report['1']['precision'], 'Recall (Yes)': report['1']['recall']})
        
        # --- Store everything in Session State ---
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.raw_df = raw_df
        st.session_state.models = models
        st.session_state.predictions = predictions_temp
        st.session_state.summary_df = pd.DataFrame(summary_data).sort_values('F1-Score (Yes)', ascending=False).reset_index(drop=True)
        st.session_state.app_loaded = True # Mark the app as loaded

# --- 3. Sidebar for Filtering ---
st.sidebar.title("Dashboard Controls")
selected_models = st.sidebar.multiselect(
    "Select models to display:",
    options=list(st.session_state.models.keys()),
    default=list(st.session_state.models.keys())
)

# --- 4. Main Title ---
st.title("📊 Bank Marketing | Model Performance Dashboard")
st.markdown("An analysis of predictive models for a telemarketing campaign.")

# --- 5. Main Logic: Filter and Display Data from Session State ---
if selected_models:
    # Filter the main summary dataframe based on the user's selection
    filtered_summary_df = st.session_state.summary_df[st.session_state.summary_df['Model'].isin(selected_models)]
    filtered_predictions = {k: v for k, v in st.session_state.predictions.items() if k in selected_models}
    
    if not filtered_summary_df.empty:
        champion_name = filtered_summary_df.iloc[0]['Model']
        
        # --- Dashboard Layout ---
        tab_perf, tab_strat, tab_interp = st.tabs(["📈 Performance Overview", "🎯 Strategic Analysis", "🧠 Model Interpretability"])

        with tab_perf:
            kpi_cards.show(filtered_summary_df)
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                performance_charts.show_ranking(filtered_summary_df)
            with col2:
                performance_charts.show_donuts(filtered_summary_df)
            with st.expander("Click to view the full performance data table"):
                st.dataframe(filtered_summary_df)

        with tab_strat:
            st.header("Strategic Simulation and Business Impact")
            if champion_name in filtered_predictions:
                strategic_charts.show_threshold_slider(filtered_predictions[champion_name]['probas'][:, 1], st.session_state.y_test, champion_name)
            
        with tab_interp:
            st.header("Understanding Model Logic")
            model_to_inspect = st.selectbox("Select a model to interpret:", options=selected_models)
            if model_to_inspect:
                interpretability_charts.show_subgroup_errors(st.session_state.raw_df, st.session_state.X_test, st.session_state.y_test, st.session_state.models[model_to_inspect], model_to_inspect)
else:
    st.warning("Please select at least one model from the sidebar to display the analysis.")