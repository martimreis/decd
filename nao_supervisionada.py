import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

sns.set_theme(style="whitegrid")
os.makedirs("output", exist_ok=True)

# 1. Ler dados
df = pd.read_csv("CVD_cleaned.csv")
df_clean = df.drop_duplicates().copy()

# 2. Preparação para clustering
num_cols = df_clean.select_dtypes(exclude="object").columns
df_zscore = df_clean.copy()
df_zscore[num_cols] = StandardScaler().fit_transform(df_zscore[num_cols])

X = df_zscore[num_cols].dropna()

# Amostras para tornar o processo viável
X_model = X.sample(n=min(5000, len(X)), random_state=42)
X_plot = X.sample(n=min(2000, len(X)), random_state=42)

# ==================================================
# 1) K-MEANS
# ==================================================
k_values = range(2, 11)
inertias = []
silhouettes = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_model)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_model, labels))

best_k = list(k_values)[int(np.argmax(silhouettes))]

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels_model = kmeans.fit_predict(X_model)

# PCA para visualização
pca = PCA(n_components=2, random_state=42)
X_plot_pca = pca.fit_transform(X_plot)
kmeans_plot = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_model)
# usar predição no conjunto de plot via centroides não é direto com subsets diferentes,
# então ajusta-se um KMeans também no subset de plot para visualização
kmeans_plot_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_plot_labels = kmeans_plot_model.fit_predict(X_plot)
X_plot_pca = pca.fit_transform(X_plot)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method - K-Means")
plt.xlabel("k")
plt.ylabel("Inertia")

plt.subplot(1, 2, 2)
plt.plot(list(k_values), silhouettes, marker="o", color="green")
plt.title("Silhouette Score - K-Means")
plt.xlabel("k")
plt.ylabel("Score")

plt.tight_layout()
plt.savefig("output/07_kmeans_metrics.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(X_plot_pca[:, 0], X_plot_pca[:, 1], c=kmeans_plot_labels, cmap="tab10", s=18, alpha=0.8)
plt.title(f"K-Means em PCA 2D (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig("output/08_kmeans_clusters_pca.png", dpi=200, bbox_inches="tight")
plt.show()

# ==================================================
# 2) CLUSTERING HIERÁRQUICO
# ==================================================
X_dendro = X.sample(n=min(1000, len(X)), random_state=42)

Z = linkage(X_dendro, method="ward")
plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True, color_threshold=None)
plt.title("Dendrograma - Clustering Hierárquico (Ward)")
plt.xlabel("Observações")
plt.ylabel("Distância")
plt.tight_layout()
plt.savefig("output/09_dendrogram.png", dpi=200, bbox_inches="tight")
plt.show()

hier_k = best_k
hier = AgglomerativeClustering(n_clusters=hier_k, linkage="ward")
hier_labels_model = hier.fit_predict(X_model)

hier_plot = AgglomerativeClustering(n_clusters=hier_k, linkage="ward")
hier_plot_labels = hier_plot.fit_predict(X_plot)

plt.figure(figsize=(9, 6))
plt.scatter(X_plot_pca[:, 0], X_plot_pca[:, 1], c=hier_plot_labels, cmap="tab10", s=18, alpha=0.8)
plt.title(f"Clustering Hierárquico em PCA 2D (k={hier_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig("output/10_hierarchical_clusters_pca.png", dpi=200, bbox_inches="tight")
plt.show()

# ==================================================
# 3) DBSCAN
# ==================================================
eps_values = [0.5, 0.8, 1.0, 1.2]
min_samples = 10

best_dbscan = None
best_dbscan_labels = None
best_dbscan_score = -np.inf
best_eps = None

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_model)

    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    if n_clusters < 2:
        continue

    sil = silhouette_score(X_model[mask], labels[mask])

    if sil > best_dbscan_score:
        best_dbscan_score = sil
        best_dbscan = db
        best_dbscan_labels = labels
        best_eps = eps

if best_eps is None:
    best_eps = 1.0
    best_dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
    best_dbscan_labels = best_dbscan.fit_predict(X_model)

db_plot = DBSCAN(eps=best_eps, min_samples=min_samples)
db_plot_labels = db_plot.fit_predict(X_plot)

plt.figure(figsize=(9, 6))
plt.scatter(X_plot_pca[:, 0], X_plot_pca[:, 1], c=db_plot_labels, cmap="tab10", s=18, alpha=0.8)
plt.title(f"DBSCAN em PCA 2D (eps={best_eps}, min_samples={min_samples})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster (-1 = ruído)")
plt.tight_layout()
plt.savefig("output/11_dbscan_clusters_pca.png", dpi=200, bbox_inches="tight")
plt.show()

# ==================================================
# 4) MÉTRICAS
# ==================================================
def safe_metrics(Xdata, labels):
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return np.nan, np.nan, np.nan
    return (
        silhouette_score(Xdata[mask], labels[mask]),
        davies_bouldin_score(Xdata[mask], labels[mask]),
        calinski_harabasz_score(Xdata[mask], labels[mask])
    )

kmeans_sil, kmeans_db, kmeans_ch = safe_metrics(X_model, kmeans_labels_model)
hier_sil, hier_db, hier_ch = safe_metrics(X_model, hier_labels_model)
db_sil, db_db, db_ch = safe_metrics(X_model, best_dbscan_labels)

metrics = pd.DataFrame([
    ["K-Means", best_k, np.nan, kmeans_sil, kmeans_db, kmeans_ch],
    ["Hierárquico", hier_k, np.nan, hier_sil, hier_db, hier_ch],
    ["DBSCAN", np.nan, best_eps, db_sil, db_db, db_ch],
], columns=["Método", "k", "eps", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"])

metrics.to_csv("output/12_clustering_metrics.csv", index=False)
print(metrics)
print(f"Best k (K-Means): {best_k}")
print(f"Best eps (DBSCAN): {best_eps}")

# ==================================================
# 4.1) TABELA VISUAL DAS MÉTRICAS
# ==================================================
metrics_display = metrics.copy()
metrics_display["k"] = metrics_display["k"].apply(lambda x: "-" if pd.isna(x) else int(x))
metrics_display["eps"] = metrics_display["eps"].apply(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
metrics_display["Silhouette"] = metrics_display["Silhouette"].apply(lambda x: f"{x:.4f}")
metrics_display["Davies-Bouldin"] = metrics_display["Davies-Bouldin"].apply(lambda x: f"{x:.4f}")
metrics_display["Calinski-Harabasz"] = metrics_display["Calinski-Harabasz"].apply(lambda x: f"{x:.2f}")

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.axis("off")

table = ax.table(
    cellText=metrics_display.values,
    colLabels=metrics_display.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#2f5d62")
    elif row % 2 == 1:
        cell.set_facecolor("#f3f6f6")
    else:
        cell.set_facecolor("#ffffff")
    cell.set_edgecolor("#c9d1d3")

plt.tight_layout()
plt.savefig("output/12_clustering_metrics_table.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()