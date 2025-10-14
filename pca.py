import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE




def pca_tsne_plots():
print("Loading methylation data...")


# Load data
df_v1 = pd.read_csv("/home/ubuntu/hackathon_data/GSE286313_MatrixProcessed_GPL21145.csv", nrows=1000)
df_v2 = pd.read_csv("/home/ubuntu/hackathon_data/GSE286313_MatrixProcessed_GPL33022.csv", nrows=1000)


# Clean columns
df_v1.columns = df_v1.columns.str.strip('"')
df_v2.columns = df_v2.columns.str.strip('"')


df_v1 = df_v1.set_index("ID_REF")
df_v2 = df_v2.set_index("ID_REF")


# Filter out Detection Pval columns
v1_meth_cols = [col for col in df_v1.columns if "Detection Pval" not in col]
v2_meth_cols = [col for col in df_v2.columns if "Detection Pval" not in col]


df_v1_meth = df_v1[v1_meth_cols]
df_v2_meth = df_v2[v2_meth_cols]


# Convert to numeric
df_v1_meth = df_v1_meth.apply(pd.to_numeric, errors='coerce').fillna(0)
df_v2_meth = df_v2_meth.apply(pd.to_numeric, errors='coerce').fillna(0)


# Use only common probes
common_probes = df_v1_meth.index.intersection(df_v2_meth.index)
if len(common_probes) == 0:
raise ValueError("No common probes between platforms")


df_v1_common = df_v1_meth.loc[common_probes].T
df_v2_common = df_v2_meth.loc[common_probes].T


df_v1_common['Platform'] = 'EPIC v1'
df_v2_common['Platform'] = 'EPIC v2'


df_all = pd.concat([df_v1_common, df_v2_common])
X = df_all.drop('Platform', axis=1)
y = df_all['Platform']


# === PCA ===
print("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.figure(figsize=(10, 8))
for platform in y.unique():
idx = y == platform
plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=platform, alpha=0.7)


plt.title("PCA of Methylation Data")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.legend()
plt.tight_layout()
plt.savefig("/home/ubuntu/pca_combined_platforms.png", dpi=300)
plt.close()
print("Saved PCA plot: pca_combined_platforms.png")


# === t-SNE ===
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
X_tsne = tsne.fit_transform(X)


plt.figure(figsize=(10, 8))
for platform in y.unique():
idx = y == platform
plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=platform, alpha=0.7)


plt.title("t-SNE of Methylation Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
pca_tsne_plots()
