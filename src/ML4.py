# ======================================================
# Rede Neural com dados bem comportados
# Prevendo aprovação de alunos + GRÁFICOS DIDÁTICOS
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# 1. Criando um dataset mais suave e realista
# ------------------------------------------------------
dados = pd.DataFrame({
    'Horas_Estudo':     [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8],
    'Frequencia':       [40,50,55,60,65,70,80,90,45,55,60,70,75,85,95],
    'Passou':           [0,0,0,0,1,1,1,1,0,0,0,1,1,1,1]
})

X = dados[['Horas_Estudo','Frequencia']]
y = dados['Passou']

# ------------------------------------------------------
# 2. Padronização
# ------------------------------------------------------
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# ------------------------------------------------------
# 3. Treino e teste
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42
)

# ------------------------------------------------------
# 4. Estrutura da rede neural (MESMA)
# ------------------------------------------------------
modelo = Sequential([
    Dense(6, activation='relu', input_shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ------------------------------------------------------
# 5. Compilação (MESMA)
# ------------------------------------------------------
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------------------
# 6. Treinamento
# ------------------------------------------------------
hist = modelo.fit(X_train, y_train, epochs=150, verbose=0)

# ------------------------------------------------------
# 7. Avaliação
# ------------------------------------------------------
loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
print(f"Acurácia no teste: {acc:.2f}")

# ------------------------------------------------------
# 8. Previsão
# ------------------------------------------------------
novo = np.array([[5, 75]])
novo_prep = scaler.transform(novo)
prob = modelo.predict(novo_prep, verbose=0)[0][0]

print(f"Probabilidade de passar: {prob:.2f}")
print("Passa!" if prob >= 0.5 else "Não passa!")

# ======================================================
# GRÁFICO 1 — Dispersão dos dados reais
# ======================================================

plt.figure(figsize=(7,5))
plt.scatter(dados[dados.Passou==1].Horas_Estudo,
            dados[dados.Passou==1].Frequencia,
            color='green', label='Passou')

plt.scatter(dados[dados.Passou==0].Horas_Estudo,
            dados[dados.Passou==0].Frequencia,
            color='red', label='Não passou')

plt.title("Distribuição dos Alunos\n(Horas Estudadas x Frequência)")
plt.xlabel("Horas Estudadas")
plt.ylabel("Frequência (%)")
plt.grid(True)
plt.legend()
plt.show()

# ======================================================
# GRÁFICO 2 — Curva de perda (loss curve)
# ======================================================

plt.figure(figsize=(7,5))
plt.plot(hist.history['loss'])
plt.title("Curva de Treinamento\n(Evolução da Função de Perda)")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ======================================================
# GRÁFICO 3 — Comparando valores reais x previstos
# ======================================================

y_pred = (modelo.predict(X_norm) > 0.5).astype(int).flatten()

plt.figure(figsize=(10,5))
plt.plot(y.values, 'o-', label='Real')
plt.plot(y_pred, 's--', label='Previsto')
plt.title("Comparação: Real x Previsto")
plt.xlabel("Índice do Aluno")
plt.ylabel("Passou (1) / Não passou (0)")
plt.legend()
plt.grid()
plt.show()

# ======================================================
# GRÁFICO 4 — Fronteira de decisão
# ======================================================

# criando grade
h1_min, h1_max = X_norm[:,0].min()-0.5, X_norm[:,0].max()+0.5
h2_min, h2_max = X_norm[:,1].min()-0.5, X_norm[:,1].max()+0.5

xx, yy = np.meshgrid(np.linspace(h1_min, h1_max, 200),
                     np.linspace(h2_min, h2_max, 200))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(grid, verbose=0).reshape(xx.shape)

plt.figure(figsize=(7,5))
plt.contourf(xx, yy, Z, cmap='RdYlGn', alpha=0.4)

plt.scatter(X_norm[y==1][:,0], X_norm[y==1][:,1],
            color='green', label="Passou")
plt.scatter(X_norm[y==0][:,0], X_norm[y==0][:,1],
            color='red', label="Não passou")

plt.title("Fronteira de Decisão da Rede Neural")
plt.xlabel("Horas (Normalizado)")
plt.ylabel("Frequência (Normalizado)")
plt.legend()
plt.grid()
plt.show()
