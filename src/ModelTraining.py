import joblib
import TransCalc as transCalc
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Constants import MODELS_DIR

PARTIAL_DATA = False
ITERATIONS = 50
MODEL_PATH = f"{MODELS_DIR}\\DMTNN_{ITERATIONS}_{"PARTIAL" if PARTIAL_DATA else "FULL"}"

dataSet = transCalc.readData(partial=PARTIAL_DATA)

X = dataSet[['Hatch', 'Charge', 'Species', 'Level', 'Clone']]
y = dataSet['Experience']

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=ITERATIONS,
    verbose=1,
    validation_split=0.1,
)

model.save(f"{MODEL_PATH}.keras")
joblib.dump((x_scaler, y_scaler), f"{MODEL_PATH}.scalers")

print(f"\nModel Training Complete, saved at: {MODEL_PATH}")
