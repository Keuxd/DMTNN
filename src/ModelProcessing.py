import joblib
import numpy as np
from keras.models import load_model

MODEL_FILE_NAME = "D:\\dmtnn"

model = load_model(f"{MODEL_FILE_NAME}.keras")
x_scaler, y_scaler = joblib.load(f"{MODEL_FILE_NAME}.scalers")

model.summary()

TARGET_EXPERIENCE = 25.0

TARGET_SCALED = y_scaler.transform(np.array([[TARGET_EXPERIENCE]]))

MAX_LEVEL = 120
MAX_REINFORCEMENT = 60

best_combination = None
min_positive_difference = float('inf')

search_space = []
for level in range(1, MAX_LEVEL + 1):
    for reinforcement in range(0, MAX_REINFORCEMENT + 1):
        search_space.append([level, reinforcement])

search_space = np.array(search_space)

search_space_scaled = x_scaler.transform(search_space)

predicted_scaled = model.predict(search_space_scaled, verbose=0)

predicted_experience = y_scaler.inverse_transform(predicted_scaled)

for i in range(len(search_space)):
    level, reinforcement = search_space[i]
    experience = predicted_experience[i][0]
    print(experience)

    if experience >= TARGET_EXPERIENCE:
        difference = experience - TARGET_EXPERIENCE
        
        if difference < min_positive_difference:
            min_positive_difference = difference
            best_combination = (level, reinforcement, experience)

if best_combination:
    level, reinforcement, experience = best_combination
    
    print("---")
    print("Optimal Combination Found:")
    print(f"Level: **{level}**")
    print(f"Reinforcement: **{reinforcement}**")
    print(f"Predicted Experience: **{experience:.2f}** (Difference: {min_positive_difference:.2f})")
    print("---")
else:
    print(f"Could not find a combination that yields an Experience of {TARGET_EXPERIENCE:.2f} or higher within the tested range.")


