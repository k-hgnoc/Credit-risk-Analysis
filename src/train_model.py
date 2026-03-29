import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ===== 1. PATH =====
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'processed', 'clean_loan.csv')

print(f"--- Đang nạp dữ liệu từ: {data_path} ---")

if not os.path.exists(data_path):
    print("❌ Không tìm thấy dữ liệu!")
    exit()

# ===== 2. LOAD DATA =====
df = pd.read_csv(data_path, low_memory=False)

# ===== 3. TIỀN XỬ LÝ =====
print("--- Đang xử lý dữ liệu ---")

# term: "36 months" → 36
if 'term' in df.columns and df['term'].dtype == 'object':
    df['term'] = df['term'].str.extract('(\d+)').astype(float)

# encode categorical
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str).fillna('Unknown'))

# fill missing
df = df.fillna(df.median(numeric_only=True))

# ===== 4. CHỌN FEATURE (QUAN TRỌNG) =====
features = [
    'loan_amnt','term','int_rate','installment',
    'annual_inc','dti','fico_range_low','fico_range_high',
    'emp_length','home_ownership'
]

# kiểm tra đủ cột chưa
missing = [col for col in features if col not in df.columns]
if missing:
    print("❌ Thiếu cột:", missing)
    exit()

X = df[features]
y = df['loan_status']

# ===== 5. TRAIN TEST =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 6. TRAIN MODEL =====
print("--- Training Logistic Regression ---")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===== 7. EVALUATE =====
y_pred = model.predict(X_test)

print("\n" + "="*30)
print("KẾT QUẢ:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))
print("="*30)

# ===== 8. SAVE MODEL =====
model_path = os.path.join(current_dir, '..', 'models', 'logistic_model.pkl')
feature_path = os.path.join(current_dir, '..', 'models', 'feature_names.pkl')

os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(model, model_path)
joblib.dump(features, feature_path)
print(f"✅ Model saved: {model_path}")
print(f"✅ Features saved: {feature_path}")

# ===== SAVE MODEL =====
model_path = os.path.join(current_dir, '..', 'models', 'logistic_model.pkl')
feature_path = os.path.join(current_dir, '..', 'models', 'feature_names.pkl')
background_path = os.path.join(current_dir, '..', 'models', 'background.pkl')

os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(model, model_path)
joblib.dump(features, feature_path)

# 🔥 LƯU BACKGROUND (QUAN TRỌNG NHẤT)
background = X_train.sample(100, random_state=42)
joblib.dump(background, background_path)

print(f"✅ Model saved: {model_path}")
print(f"✅ Features saved: {feature_path}")
print(f"✅ Background saved: {background_path}")