# ======================================================
# RED NEURONAL – REGRESIÓN DE VENTAS_PROM_MENSUALES
# Autores: Katya Iman & Angelo Gomez
# Fecha: Diciembre 2024
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Cargar dataset
# -----------------------------
df = pd.read_csv("dataset_mipymes_100k_nombres.csv")
print("Shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Definir variables
# -----------------------------
target = "ventas_prom_mensuales"

numeric_features = [
    "nro_trabajadores",
    "antiguedad_anios",
    "saldo_credito_miles",
    "activos_miles",
    "gastos_miles",
    "exporta_flag",
    "moroso_flag",
    "anio"
]

categorical_features = ["rubro", "tamano", "departamento"]

X = df[numeric_features + categorical_features]
y = df[target].values

# -----------------------------
# 3. Preprocesamiento (scaling + one-hot)
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)
print("Shape X procesado:", X_processed.shape)

# -----------------------------
# 4. Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Definir modelo de red neuronal
# -----------------------------
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # salida continua (regresión)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

model.summary()

# -----------------------------
# 6. Entrenamiento con EarlyStopping
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 7. Curva de pérdida
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.title("Curva de pérdida – Red neuronal (ventas)")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 8. Evaluación en test
# -----------------------------
y_pred = model.predict(X_test).ravel()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# -----------------------------
# 9. Gráfico Real vs Predicho
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=5, alpha=0.5)
plt.xlabel("Ventas reales (k soles/mes)")
plt.ylabel("Ventas predichas")
plt.title("Real vs Predicho – Red neuronal")

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.grid(True)
plt.show()
