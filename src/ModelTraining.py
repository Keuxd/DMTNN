import joblib
import TransCalc as transCalc
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Enums import HatchLevel, SpeciesMultiplier, ChargeMultiplier

MODEL_FILE_NAME = "D:\\dmtnn"

dataSet = transCalc.readDataSet(HatchLevel.LEVEL5, ChargeMultiplier.REGULAR_CHARGE, SpeciesMultiplier.DIFFERENT_SPECIES)

X = dataSet[['Level','Reinforcement']]
y = dataSet['Experience']

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=42
)

model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear') 
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
    epochs=200,
    batch_size=4,
    verbose=1,
    validation_split=0.1
)

model.save(f"{MODEL_FILE_NAME}.keras")
joblib.dump((x_scaler, y_scaler), f"{MODEL_FILE_NAME}.scalers")

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Teste Accuracy: {acc:.2f}")

print(f"\nModel Training Complete, saved at: {MODEL_FILE_NAME}")