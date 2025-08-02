import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report

def show(predictions, y_test):
    """Creates the performance comparison tab."""
    st.header("Comparative Model Performance")

    if not predictions:
        st.warning("No models selected in the sidebar.")
        return

    # --- 1. Summary Table ---
    summary_data = []
    for model_name, preds in predictions.items():
        report = classification_report(y_test, (preds['probas'][:, 1] > 0.5).astype(int), output_dict=True, zero_division=0)
        summary_data.append({
            'Model': model_name,
            'F1-Score (Yes)': report['1']['f1-score'],
            'Precision (Yes)': report['1']['precision'],
            'Recall (Yes)': report['1']['recall']
        })
    summary_df = pd.DataFrame(summary_data).sort_values('F1-Score (Yes)', ascending=False)
    st.dataframe(summary_df)

    # --- 2. Interactive Bar Chart ---
    st.subheader("Metric Comparison")
    df_melted = summary_df.melt(id_vars='Model', value_vars=['Precision (Yes)', 'Recall (Yes)', 'F1-Score (Yes)'])
    fig = px.bar(df_melted, x='Model', y='value', color='variable', barmode='group', text='value')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_range=[0,1], xaxis_tickangle=-45, title_x=0.5, legend_title_text='Metrics')
    st.plotly_chart(fig, use_container_width=True)