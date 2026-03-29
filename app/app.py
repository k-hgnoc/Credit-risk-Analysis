import streamlit as st
import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt

# ===== CONFIG =====
st.set_page_config(page_title="Credit Risk", layout="wide")

# ===== PATH =====
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, '..', 'models', 'logistic_model.pkl')
bg_path = os.path.join(current_dir, '..', 'models', 'background.pkl')

# ===== LOAD MODEL =====
if not os.path.exists(model_path):
    st.error("❌ Không tìm thấy model! Hãy chạy train_model.py trước")
    st.stop()

model = joblib.load(model_path)

# ===== LOAD BACKGROUND (QUAN TRỌNG) =====
if not os.path.exists(bg_path):
    st.error("❌ Không tìm thấy background data! Hãy train lại model")
    st.stop()

background = joblib.load(bg_path)

# ===== UI =====
st.title("💳 Credit Risk Dashboard")
st.subheader("Nhập thông tin khách hàng")

loan_amnt = st.number_input("Loan Amount", value=10000)
term = st.selectbox("Term", [36, 60])
int_rate = st.number_input("Interest Rate", value=10.0)
installment = st.number_input("Installment", value=300.0)
annual_inc = st.number_input("Annual Income", value=50000)
dti = st.number_input("Debt-to-Income Ratio", value=15.0)
fico_low = st.number_input("FICO Low", value=650)
fico_high = st.number_input("FICO High", value=700)
emp_length = st.number_input("Employment Length", value=5)
home = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}

# ===== PREDICT + XAI =====
if st.button("Predict"):

    data = pd.DataFrame([[
        loan_amnt, term, int_rate, installment,
        annual_inc, dti, fico_low, fico_high,
        emp_length, home_map[home]
    ]], columns=[
        'loan_amnt','term','int_rate','installment',
        'annual_inc','dti','fico_range_low','fico_range_high',
        'emp_length','home_ownership'
    ])

    # ===== PREDICT =====
    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    st.write(f"📊 Risk Score: {prob:.2f}")
    st.progress(prob)

    if prob > 0.7:
        st.error("🔴 Risk cao")
    elif prob > 0.4:
        st.warning("🟡 Risk trung bình")
    else:
        st.success("🟢 Risk thấp")

    # ===== XAI (DÙNG BACKGROUND THẬT) =====
    st.subheader("🔍 Giải thích dự đoán")

    try:
        # 👉 dùng background từ train
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(data)

        shap_df = pd.DataFrame({
            "Feature": data.columns,
            "Impact": shap_values[0]
        })

        # scale cho dễ nhìn
        shap_df["Impact"] = shap_df["Impact"] * 100

        # sort cho đẹp
        shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=True)

        st.dataframe(shap_df)

        # ===== BAR CHART =====
        st.subheader("📊 Mức độ ảnh hưởng")

        fig, ax = plt.subplots()
        ax.barh(shap_df["Feature"], shap_df["Impact"])

        ax.set_xlabel("Impact (scaled)")
        ax.set_title("SHAP Explanation")
        ax.invert_yaxis()

        st.pyplot(fig)

    except Exception as e:
        st.error("❌ SHAP lỗi")
        st.write(e)