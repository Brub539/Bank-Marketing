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

# --- 2. Data & Model Loading (Cached) ---
X_test, y_test, raw_df, models = load_all()

# --- 3. Initialize Session State ---
# This block runs only once at the beginning of the session.
if 'predictions_generated' not in st.session_state:
    st.session_state.predictions_generated = False
    st.session_state.predictions = {}
    st.session_state.summary_df = pd.DataFrame()

# --- 4. Sidebar Controls ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Configure your analysis below and click the button to run.")

with st.sidebar.form("controls_form"):
    model_names = list(models.keys())
    selected_models = st.multiselect(
        "Select models to compare:",
        options=model_names,
        default=model_names
    )
    # This button will trigger the one-time prediction generation
    run_button = st.form_submit_button("🚀 Run Analysis")

# --- 5. Main Title ---
st.title("📊 Bank Marketing | Model Performance Dashboard")
st.markdown("An analysis of predictive models for a telemarketing campaign.")

# --- 6. Main Logic: Generate Predictions ONCE when button is clicked ---
if run_button:
    # This block runs only when the "Run Analysis" button is clicked.
    with st.spinner("Generating predictions for selected models..."):
        predictions_temp = {}
        summary_data_temp = []
        for model_name in selected_models:
            model = models[model_name]
            is_neural_net = 'Tabnet' in model_name or isinstance(model, tf.keras.Model)
            X_test_input = X_test.to_numpy() if is_neural_net else X_test
            
            # Get probabilities
            if isinstance(model, tf.keras.Model):
                p_yes = model.predict(X_test_input).ravel()
                probas = np.vstack((1 - p_yes, p_yes)).T
            else:
                probas = model.predict_proba(X_test_input)
            predictions_temp[model_name] = {'probas': probas}

            # Get metrics
            report = classification_report(y_test, (probas[:, 1] > 0.5).astype(int), output_dict=True, zero_division=0)
            summary_data_temp.append({'Model': model_name, 'F1-Score (Yes)': report['1']['f1-score'], 'Precision (Yes)': report['1']['precision'], 'Recall (Yes)': report['1']['recall']})
        
        # --- Store results in Session State ---
        st.session_state.predictions = predictions_temp
        st.session_state.summary_df = pd.DataFrame(summary_data_temp).sort_values('F1-Score (Yes)', ascending=False).reset_index(drop=True)
        st.session_state.predictions_generated = True

# --- 7. Display Dashboard if results exist in Session State ---
if st.session_state.predictions_generated:
    # Get data from session state - this is fast, no recalculation needed!
    summary_df = st.session_state.summary_df
    predictions = st.session_state.predictions
    
    if summary_df.empty:
        st.warning("No models were successfully analyzed. Please check your selections and run the analysis.")
    else:
        champion_name = summary_df.loc[0, 'Model']
        
        # Define the tabs
        tab_perf, tab_strat, tab_interp = st.tabs(["📈 Performance Overview", "🎯 Strategic Analysis", "🧠 Model Interpretability"])

        with tab_perf:
            kpi_cards.show(summary_df)
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                performance_charts.show_ranking(summary_df)
            with col2:
                performance_charts.show_donuts(summary_df)
            with st.expander("Click to view the full performance data table"):
                st.dataframe(summary_df)

        with tab_strat:
            st.header("Strategic Simulation and Business Impact")
            # The slider now only re-runs this small part of the code, not the predictions.
            strategic_charts.show_threshold_slider(predictions[champion_name]['probas'][:, 1], y_test, champion_name)
            
        with tab_interp:
            st.header("Understanding Model Logic")
            # The selectbox now feels instant because it's not re-running predictions.
            model_to_inspect = st.selectbox("Select a model to interpret:", options=list(predictions.keys()))
            if model_to_inspect:
                interpretability_charts.show_subgroup_errors(raw_df, X_test, y_test, models[model_to_inspect], model_to_inspect)
else:
    st.info("Welcome! Please select the models you wish to analyze in the sidebar and click 'Run Analysis'.")