import joblib
import numpy as np
from keras.models import load_model

MODEL_FILE_NAME = "D:\\dmtnn"

model = load_model(f"{MODEL_FILE_NAME}.keras")
x_scaler, y_scaler = joblib.load(f"{MODEL_FILE_NAME}.scalers")

level = 50
reinforcement = 7

input_data = np.array([[level, reinforcement]])

scaled_input = x_scaler.transform(input_data)

scaled_pred = model.predict(scaled_input)

predicted_xp = y_scaler.inverse_transform(scaled_pred)[0][0]

print(f"Predicted Experience for Level={level}, Reinforcement={reinforcement}: {predicted_xp:.2f}")
