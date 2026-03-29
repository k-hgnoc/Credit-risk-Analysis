import os
import joblib

# lấy đường dẫn file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# build path tới model
model_path = os.path.join(current_dir, '..', 'models', 'logistic_model.pkl')

print("Loading model from:", model_path)

model = joblib.load(model_path)