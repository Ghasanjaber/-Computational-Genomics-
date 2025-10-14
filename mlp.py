import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed methylation matrix (100 probes x N samples)
# Make sure the files are in the correct format: rows = probes, columns = samples
v1 = pd.read_csv('GSE286313_MatrixProcessed_GPL21145.csv', index_col=0)
v2 = pd.read_csv('GSE286313_MatrixProcessed_GPL33022.csv', index_col=0)

# Intersect on probes
common_probes = list(set(v1.index) & set(v2.index))
v1 = v1.loc[common_probes]
v2 = v2.loc[common_probes]

# Subsample 100 probes for computational ease (you can change this)
selected_probes = np.random.choice(common_probes, 100, replace=False)
v1 = v1.loc[selected_probes]
v2 = v2.loc[selected_probes]

# Transpose and label
v1 = v1.T
v2 = v2.T
v1['label'] = 0  # EPIC v1
v2['label'] = 1  # EPIC v2

# Merge
df = pd.concat([v1, v2])

# Split
X = df.drop(columns=['label']).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build MLP
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------
# Identify Most Important Probes (First Layer)
# -------------------------------------------

# Get first layer weights (shape: [n_hidden, n_features])
first_layer_weights = clf.coefs_[0]  # shape = (100 features, 50 neurons)

# Aggregate absolute weights across neurons
importance_scores = np.mean(np.abs(first_layer_weights), axis=1)

# Map back to probe names
probe_names = df.drop(columns='label').columns
probe_importance = pd.Series(importance_scores, index=probe_names)
top_probes = probe_importance.sort_values(ascending=False).head(20)

# Plot top probes
plt.figure(figsize=(12, 6))
sns.barplot(x=top_probes.values, y=top_probes.index, palette='viridis')
plt.title("Top 20 Most Influential Probes (MLP First Layer Weights)")
plt.xlabel("Mean Absolute Weight")
plt.tight_layout()
plt.savefig("top_probes_mlp.png")
plt.show()

# Save top 100 probes for later use
top_100_probes = probe_importance.sort_values(ascending=False).head(100)
top_100_probes.to_csv("top_100_mlp_probes.csv")

print("\nTop probes saved to 'top_100_mlp_probes.csv'")
