import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def show_ranking(df):
    """Creates a horizontal bar chart to rank all models by F1-Score."""
    st.subheader("Model F1-Score Ranking")
    fig = px.bar(df, x='F1-Score (Yes)', y='Model', orientation='h', text='F1-Score (Yes)', template='plotly_dark')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title=None, height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_donuts(df):
    """Creates donut charts for Precision and Recall of the champion model."""
    if df.empty: return
    
    st.subheader(f"🥇 Champion Breakdown: {df.iloc[0]['Model']}")
    champion = df.iloc[0]
    cols = st.columns(2)
    
    with cols[0]:
        fig = go.Figure(go.Pie(values=[champion['Precision (Yes)'], 1 - champion['Precision (Yes)']], labels=['Correct', 'Incorrect'], hole=.6, marker_colors=['#00cc96', '#333']))
        fig.update_layout(title_text='Precision', showlegend=False, annotations=[dict(text=f"{champion['Precision (Yes)']:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)], template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        fig = go.Figure(go.Pie(values=[champion['Recall (Yes)'], 1 - champion['Recall (Yes)']], labels=['Found', 'Missed'], hole=.6, marker_colors=['#636efa', '#333']))
        fig.update_layout(title_text='Recall', showlegend=False, annotations=[dict(text=f"{champion['Recall (Yes)']:.1%}", x=0.5, y=0.5, font_size=20, showarrow=False)], template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)