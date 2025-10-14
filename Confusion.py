# hackathon_visuals.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === SECTION 1: Confusion Matrices and Misclassification Rates ===

def plot_classification_results():
    labels = ["EPIC v1", "EPIC v2"]
    
    # Simulated confusion matrices (normalized)
    cm_mean = np.array([[0.92, 0.08], [0.08, 0.92]])
    cm_mlp = np.array([[0.96, 0.04], [0.04, 0.96]])

    # Misclassification rates
    misclass_mean = [0.08, 0.08]
    misclass_mlp = [0.04, 0.04]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Final Results for Hackathon (Example)", fontsize=14)

    # (a) Mean Classifier Confusion Matrix
    sns.heatmap(cm_mean, annot=True, fmt=".2f", cmap="Blues", ax=axs[0], cbar=False)
    axs[0].set_title("(a) Confusion Matrix\nMean Classifier")
    axs[0].set_xlabel("Predicted label")
    axs[0].set_ylabel("True label")
    axs[0].set_xticklabels(labels)
    axs[0].set_yticklabels(labels)

    # (b) Misclassification Rates
    bar_width = 0.35
    x = np.arange(len(labels))
    axs[1].bar(x - bar_width/2, misclass_mean, bar_width, label='Mean Classifier')
    axs[1].bar(x + bar_width/2, misclass_mlp, bar_width, label='MLP Classifier')
    axs[1].set_title("(b) Misclassification Rates")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_ylim(0, 0.15)
    axs[1].set_ylabel("Misclassification Rate")
    axs[1].legend()

    for i in range(len(labels)):
        axs[1].text(x[i] - bar_width/2, misclass_mean[i] + 0.005, f"{misclass_mean[i]:.2f}", ha='center')
        axs[1].text(x[i] + bar_width/2, misclass_mlp[i] + 0.005, f"{misclass_mlp[i]:.2f}", ha='center')

    # (c) MLP Classifier Confusion Matrix
    sns.heatmap(cm_mlp, annot=True, fmt=".2f", cmap="Blues", ax=axs[2], cbar=False)
    axs[2].set_title("(c) Confusion Matrix\nMLP Classifier")
    axs[2].set_xlabel("Predicted label")
    axs[2].set_xticklabels(labels)
    axs[2].set_yticklabels(labels)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("final_classification_summary.png")
    plt.show()

# === SECTION 2: Large Confusion Matrix with Metrics ===

def plot_big_confusion_with_metrics():
    # Matrix values
    TN, FP = 48900, 1325
    FN, TP = 1259, 48516
    matrix = np.array([[TN, FP], [FN, TP]])

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt=",", cmap="Blues", square=True,
                xticklabels=["Unmethylated", "Methylated"],
                yticklabels=["Unmethylated", "Methylated"], ax=ax, cbar=False)

    ax.set_title("Confusion Matrix: EPIC v1 vs v2 Methylation Status", fontsize=14)
    ax.set_xlabel("EPIC v2 Prediction", fontsize=12)
    ax.set_ylabel("EPIC v1 (Ground Truth)", fontsize=12)

    metrics_text = (f"Accuracy: {accuracy:.4f}\n"
                    f"Sensitivity: {sensitivity:.4f}\n"
                    f"Specificity: {specificity:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"F1 Score: {f1:.4f}")

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(2.05, 0.5, metrics_text, fontsize=10, va='center', bbox=props)

    plt.tight_layout()
    plt.savefig("confusion_matrix_with_metrics.png")
    plt.show()

# To run:
if __name__ == '__main__':
    plot_classification_results()
    plot_big_confusion_with_metrics()
