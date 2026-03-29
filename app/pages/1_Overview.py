import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(layout="wide")

st.title("📊 Credit Risk Overview")

# fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'clean_loan.csv')

st.write("Path:", data_path)  # debug

if not os.path.exists(data_path):
    st.error("❌ Không tìm thấy file clean_loan.csv")
    st.stop()

df = pd.read_csv(data_path)

# KPI
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Default Rate", f"{df['loan_status'].mean():.2%}")
col3.metric("Avg Income", f"{df['annual_inc'].mean():,.0f}")

# charts
col1, col2 = st.columns(2)

fig1, ax1 = plt.subplots()
sns.countplot(x='loan_status', data=df, ax=ax1)
col1.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.histplot(df['fico_range_low'], bins=30, ax=ax2)
col2.pyplot(fig2)