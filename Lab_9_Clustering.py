"""Lab 9: K-Means and Hierarchical (Agglomerative) Clustering on the
Wholesale Customers dataset (UCI)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

# 1. Load dataset
df = pd.read_csv("Wholesale_customers_data.csv")
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())

# 2. Pre-processing: use the six spending features, log-transform (skewed) and scale
features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X = StandardScaler().fit_transform(np.log1p(df[features]))

# 3. K-Means: elbow (inertia) and silhouette for k = 2..10
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"k = {k:2d}   inertia = {km.inertia_:8.2f}   silhouette = {sil[-1]:.4f}")
best_k = list(ks)[int(np.argmax(sil))]
print("\nBest k by silhouette score:", best_k)

# 4. Final K-Means model using the best k from silhouette analysis
K = best_k
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
df["KMeans"] = km.labels_
print("\nK-Means (k=%d) silhouette: %.4f" % (K, silhouette_score(X, km.labels_)))
print("Cluster sizes:\n", df["KMeans"].value_counts().sort_index())
print("\nCluster profile (mean spending):\n", df.groupby("KMeans")[features].mean().round(0))

# 5. Hierarchical clustering: compare linkage methods
print("\nHierarchical clustering (k=%d):" % K)
for link in ["ward", "complete", "average", "single"]:
    h = AgglomerativeClustering(n_clusters=K, linkage=link).fit(X)
    print(f"  {link:9s} silhouette = {silhouette_score(X, h.labels_):.4f}")
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X)
df["Hierarchical"] = hc.labels_
print("\nHierarchical (ward) cluster sizes:\n", df["Hierarchical"].value_counts().sort_index())

# 6. Compare the two algorithms and the known 'Channel' column
print("\nAdjusted Rand Index (K-Means vs Hierarchical): %.4f" % adjusted_rand_score(df["KMeans"], df["Hierarchical"]))
print("Adjusted Rand Index (K-Means vs Channel)     : %.4f" % adjusted_rand_score(df["Channel"], df["KMeans"]))

# 7. Plots (light theme)
plt.style.use("default")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_xlabel("Number of clusters (k)")
ax[0].set_ylabel("Inertia (WCSS)"); ax[0].set_title("Elbow method")
ax[1].plot(list(ks), sil, "s-", color="green"); ax[1].set_xlabel("Number of clusters (k)")
ax[1].set_ylabel("Silhouette score"); ax[1].set_title("Silhouette analysis")
plt.tight_layout(); plt.savefig("elbow_silhouette.png", dpi=150); plt.close()

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
for a, col, title in zip(ax, ["KMeans", "Hierarchical"], ["K-Means (k=%d)" % K, "Hierarchical - Ward (k=%d)" % K]):
    sc = a.scatter(Z[:, 0], Z[:, 1], c=df[col], cmap="viridis", s=14)
    a.set_title(title); a.set_xlabel("PCA component 1")
ax[0].set_ylabel("PCA component 2")
plt.tight_layout(); plt.savefig("cluster_scatter.png", dpi=150); plt.close()

plt.figure(figsize=(9, 4.5))
dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=30, leaf_rotation=90, color_threshold=0)
plt.title("Dendrogram (Ward linkage, last 30 merges)")
plt.xlabel("Cluster (number of samples)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig("dendrogram.png", dpi=150); plt.close()
