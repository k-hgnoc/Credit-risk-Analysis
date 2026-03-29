import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.title("📊 Data Analysis")

# 📌 LẤY ĐƯỜNG DẪN CHẮC CHẮN
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'clean_loan.csv')

# debug (optional)
st.write("Data path:", data_path)

# kiểm tra file tồn tại
if not os.path.exists(data_path):
    st.error("❌ Không tìm thấy file clean_loan.csv")
    st.stop()

# load data
df = pd.read_csv(data_path)

# biểu đồ target
st.subheader("Loan Status Distribution")
fig, ax = plt.subplots()
sns.countplot(x='loan_status', data=df, ax=ax)
st.pyplot(fig)

# income vs risk
st.subheader("Income vs Risk")
fig, ax = plt.subplots()
sns.boxplot(x='loan_status', y='annual_inc', data=df, ax=ax)
st.pyplot(fig)