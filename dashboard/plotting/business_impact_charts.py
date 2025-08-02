import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import f1_score

def calculate_cumulative_gain(y_true, y_probas):
    """Helper function for cumulative gains chart."""
    df = pd.DataFrame({'y_true': y_true, 'y_probas': y_probas}).sort_values('y_probas', ascending=False)
    df['cumulative_positives'] = df['y_true'].cumsum()
    total_positives = df['y_true'].sum()
    df['percentage_of_positives_captured'] = df['cumulative_positives'] / total_positives
    df['percentage_of_sample_targeted'] = np.arange(1, len(df) + 1) / len(df)
    return df['percentage_of_sample_targeted'].tolist(), df['percentage_of_positives_captured'].tolist()

def show_cumulative_gain(predictions, y_test):
    """Creates the interactive Cumulative Gains chart."""
    st.subheader("Cumulative Gains Chart")
    st.markdown("This chart shows the percentage of successful subscriptions captured by contacting the top X% of customers as ranked by each model.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Selection'))
    
    for model_name, data in predictions.items():
        x_vals, y_vals = calculate_cumulative_gain(y_test, data['probas'][:, 1])
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=model_name, mode='lines'))

    fig.update_layout(
        xaxis_title='Percentage of Sample Targeted',
        yaxis_title='Percentage of "Yes" Captured',
        height=500,
        legend_title_text='Models',
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%")
    )
    st.plotly_chart(fig, use_container_width=True)

def show_performance_stability(predictions, y_test, X_test, champion_name):
    """Plots the F1-score of the champion model over time."""
    st.subheader(f"Performance Stability for '{champion_name}'")
    st.markdown("This chart shows if the model's performance degrades on more recent data within the test set.")
    
    y_pred = (predictions[champion_name]['probas'][:, 1] > 0.5).astype(int)
    time_analysis_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}, index=X_test.index)
    
    quarters = pd.qcut(time_analysis_df.index, 4, labels=["Quarter 1 (Oldest)", "Quarter 2", "Quarter 3", "Quarter 4 (Newest)"])
    time_analysis_df['time_quarter'] = quarters
    
    f1_by_quarter = time_analysis_df.groupby('time_quarter').apply(lambda g: f1_score(g['y_true'], g['y_pred'], pos_label=1, zero_division=0))
    
    fig = px.line(x=f1_by_quarter.index, y=f1_by_quarter.values, markers=True, labels={'x': 'Time Period (in test set)', 'y': 'F1-Score (Yes)'})
    fig.update_layout(yaxis_range=[0,1])
    st.plotly_chart(fig, use_container_width=True)