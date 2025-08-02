import streamlit as st

def show(df):
    """Creates KPI metric cards for the top 3 models."""
    st.subheader("🏆 Top Model Performance")
    
    if df.empty or len(df) < 1:
        st.warning("Not enough model data to display KPIs.")
        return
        
    top_3_models = df.head(3)
    baseline_f1 = df[df['Model'].str.contains("Logistic Regression", case=False)]['F1-Score (Yes)'].iloc[0] if not df[df['Model'].str.contains("Logistic Regression", case=False)].empty else 0

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