# ============================================
# CLUSTERING K-MEANS EN MIPYMES
# Autores: Katya Iman & Angelo Gomez
# Fecha: Diciembre 2024
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. Cargar dataset
# -----------------------------
df = pd.read_csv("dataset_mipymes_100k_nombres.csv")
print("Shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Selección de variables numéricas
# -----------------------------
features_num = [
    "nro_trabajadores",
    "antiguedad_anios",
    "saldo_credito_miles",
    "activos_miles",
    "gastos_miles",
    "ventas_prom_mensuales"
]

X = df[features_num].values

# -----------------------------
# 3. Escalado de variables
# -----------------------------
X_scaled = StandardScaler().fit_transform(X)

# -----------------------------
# 4. Método del codo (elbow)
# -----------------------------
wcss = []
K_range = range(1, 21)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title("Método del codo – K-Means MIPYME")
plt.xlabel("Número de clusters (k)")
plt.ylabel("WCSS (inertia)")
plt.grid(True)
plt.show()

# -----------------------------
# 5. Entrenar K-Means con k óptimo
# -----------------------------
k_opt = 5

kmeans = KMeans(
    n_clusters=k_opt,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster_kmeans"] = cluster_labels

print(df[["nombre_empresa", "cluster_kmeans"] + features_num].head())

# -----------------------------
# 6. Silhouette score (extra)
# -----------------------------
sil_score = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette score K-Means (k={k_opt}): {sil_score:.4f}")

# -----------------------------
# 7. Gráfico 2D: Activos vs Ventas
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    df["activos_miles"],
    df["ventas_prom_mensuales"],
    c=df["cluster_kmeans"],
    s=10
)
plt.title("Clusters K-Means – Activos vs Ventas")
plt.xlabel("Activos (miles de soles)")
plt.ylabel("Ventas prom. mensuales (miles de soles)")
plt.grid(True)
plt.show()

# -----------------------------
# 8. Gráfico 3D: Trabajadores vs Activos vs Ventas
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(
    df["nro_trabajadores"],
    df["activos_miles"],
    df["ventas_prom_mensuales"],
    c=df["cluster_kmeans"],
    s=5
)

ax.set_xlabel("Nº trabajadores")
ax.set_ylabel("Activos (miles)")
ax.set_zlabel("Ventas prom. mensuales (miles)")
plt.title("Clusters K-Means – Vista 3D")
fig.colorbar(p, label="Cluster")
plt.show()