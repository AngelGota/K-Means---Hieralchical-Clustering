# =====================================================
# CLUSTERING JERÁRQUICO EN MIPYMES
# Autores: Katya Iman & Angelo Gomez
# Fecha: Diciembre 2024
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

# -----------------------------
# 1. Cargar dataset
# -----------------------------
df = pd.read_csv("dataset_mipymes_100k_nombres.csv")
print("Shape:", df.shape)

# -----------------------------
# 2. Tomar una muestra (para no matar el dendrograma)
# -----------------------------
df_sample = df.sample(1000, random_state=42)
print(df_sample.head())

# -----------------------------
# 3. Seleccionar 2 variables numéricas
# -----------------------------
features_h = ["activos_miles", "ventas_prom_mensuales"]
X_h = df_sample[features_h].values

# -----------------------------
# 4. Escalar datos
# -----------------------------
scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)

# -----------------------------
# 5. Dendrograma (Ward linkage)
# -----------------------------
plt.figure(figsize=(12, 6))
plt.title("Dendrograma – MIPYME (muestra 1000)")
plt.ylabel("Distancia")

dend = shc.dendrogram(
    shc.linkage(X_h_scaled, method="ward")
)

# Línea horizontal para sugerir número de clusters
plt.axhline(y=15, color="r", linestyle="--")
plt.show()

# -----------------------------
# 6. Agglomerative Clustering
# -----------------------------
n_clusters_h = 5

hc = AgglomerativeClustering(
    n_clusters=n_clusters_h,
    metric='euclidean',
    linkage='ward'
)

labels_h = hc.fit_predict(X_h_scaled)
df_sample["cluster_hc"] = labels_h

print(df_sample[["nombre_empresa"] + features_h + ["cluster_hc"]].head())

# -----------------------------
# 7. Visualización 2D
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    df_sample["activos_miles"],
    df_sample["ventas_prom_mensuales"],
    c=df_sample["cluster_hc"],
    s=20
)
plt.title("Clusters – Clustering jerárquico (muestra 1000)")
plt.xlabel("Activos (miles de soles)")
plt.ylabel("Ventas prom. mensuales (miles de soles)")
plt.grid(True)
plt.show()
