import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def plot_kpi_cards(df):
    """Creates KPI metric cards for the top 3 models."""
    st.subheader("🏆 Top Model Performance")
    
    # Get top 3 models and a baseline for comparison
    top_3_models = df.head(3)
    baseline_f1 = df[df['Model'].str.contains("Logistic Regression", case=False)]['F1-Score (Yes)'].iloc[0]

    cols = st.columns(3)
    for i, row in top_3_models.iterrows():
        with cols[i]:
            delta = row['F1-Score (Yes)'] - baseline_f1
            st.metric(
                label=f"#{i+1} - {row['Model']}",
                value=f"{row['F1-Score (Yes)']:.3f}",
                delta=f"{delta:.3f} vs Baseline",
                help=f"Precision: {row['Precision (Yes)']:.2f}, Recall: {row['Recall (Yes)']:.2f}"
            )

def plot_performance_ranking(df):
    """Creates a horizontal bar chart to rank all models by F1-Score."""
    st.subheader("Model F1-Score Ranking")
    fig = px.bar(
        df,
        x='F1-Score (Yes)',
        y='Model',
        orientation='h',
        text='F1-Score (Yes)',
        template='plotly_dark'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title=None, height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_precision_recall_donuts(df):
    """Creates donut charts for Precision and Recall of the champion model."""
    st.subheader(f"🥇 Champion Breakdown: {df.iloc[0]['Model']}")
    
    champion = df.iloc[0]
    cols = st.columns(2)
    
    with cols[0]:
        fig = go.Figure(go.Pie(
            values=[champion['Precision (Yes)'], 1 - champion['Precision (Yes)']],
            labels=['Correct', 'Incorrect'],
            hole=.6,
            marker_colors=['#00cc96', '#333']
        ))
        fig.update_layout(title_text='Precision', showlegend=False, annotations=[dict(text=f"{champion['Precision (Yes)']:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)], template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        fig = go.Figure(go.Pie(
            values=[champion['Recall (Yes)'], 1 - champion['Recall (Yes)']],
            labels=['Found', 'Missed'],
            hole=.6,
            marker_colors=['#636efa', '#333']
        ))
        fig.update_layout(title_text='Recall', showlegend=False, annotations=[dict(text=f"{champion['Recall (Yes)']:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)], template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

def plot_subgroup_errors(raw_df, X_test, y_test, model, model_name):
    """Creates a stacked bar chart of error types by customer job."""
    st.subheader(f"Error Analysis by Job for {model_name}")
    
    X_test_input = X_test.to_numpy() if 'Tabnet' in model_name or 'Deep Learning' in model_name else X_test
    y_pred = (model.predict_proba(X_test_input)[:, 1] > 0.5).astype(int)
    
    test_jobs = raw_df.iloc[X_test.index]['job']
    error_df = pd.DataFrame({'job': test_jobs, 'y_true': y_test, 'y_pred': y_pred})
    
    # Identify outcomes
    conditions = [
        (error_df['y_true'] == 1) & (error_df['y_pred'] == 1), # True Positive
        (error_df['y_true'] == 0) & (error_df['y_pred'] == 1), # False Positive
        (error_df['y_true'] == 1) & (error_df['y_pred'] == 0), # False Negative
    ]
    choices = ['Correctly Identified', 'Wrongly Targeted', 'Missed Opportunity']
    error_df['Outcome'] = np.select(conditions, choices, default='Correctly Ignored')
    
    outcome_by_job = error_df[error_df['Outcome'] != 'Correctly Ignored'].groupby(['job', 'Outcome']).size().reset_index(name='count')
    
    fig = px.bar(outcome_by_job, x='job', y='count', color='Outcome', title="Distribution of Prediction Outcomes by Job", template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)