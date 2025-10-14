import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def mean_classifier(train_data, train_labels, test_data):
    """
    A simple mean classifier: computes the average profile for each class 
    and assigns test samples to the nearest mean.
    """
    class_means = {}
    classes = np.unique(train_labels)

    for c in classes:
        class_means[c] = train_data[train_labels == c].mean(axis=0)

    predictions = []
    for i in range(test_data.shape[0]):
        distances = {c: np.linalg.norm(test_data[i] - class_means[c]) for c in classes}
        predicted_class = min(distances, key=distances.get)
        predictions.append(predicted_class)

    return np.array(predictions)


def run_classifier():
    print("Loading methylation data...")

    # Load V1 and V2 methylation data (use only beta-values, no Detection Pval)
    v1 = pd.read_csv("data/GSE286313_MatrixProcessed_GPL21145.csv", index_col=0)
    v2 = pd.read_csv("data/GSE286313_MatrixProcessed_GPL33022.csv", index_col=0)

    # Keep only numeric columns (remove detection p-values)
    v1 = v1.loc[:, ~v1.columns.str.contains("Detection Pval", case=False)]
    v2 = v2.loc[:, ~v2.columns.str.contains("Detection Pval", case=False)]

    # Use only common probes
    common_probes = v1.index.intersection(v2.index)
    v1 = v1.loc[common_probes]
    v2 = v2.loc[common_probes]

    # Transpose to samples x features
    v1 = v1.T
    v2 = v2.T

    # Assign labels (0 for v1, 1 for v2) — you can change this logic for disease vs. healthy if known
    v1["label"] = 0
    v2["label"] = 1

    # Merge
    data = pd.concat([v1, v2], axis=0)
    X = data.drop("label", axis=1).values
    y = data["label"].values

    print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Run Mean Classifier
    print("Running Mean Classifier...")
    y_pred = mean_classifier(X_train, y_train, X_test)

    # Evaluate
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    run_classifier()
