import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc,accuracy_score
import matplotlib.pyplot as plt
import numpy as np



def compute_accuracy(scores, labels):
    """
    Compute the threshold that maximizes accuracy and the corresponding accuracy for face verification.

    Args:
        scores (list): List of similarity scores.
        labels (list): List of ground truth labels (1 for same identity, 0 for different identity).

    Returns:
        best_threshold (float): Threshold that maximizes accuracy.
        best_accuracy (float): The highest accuracy obtained.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores,drop_intermediate=False)
    
    # Compute accuracy at each threshold
    accuracies = []
    for threshold in thresholds:
        predictions = np.array(scores) >= threshold
        accuracy = accuracy_score(labels, predictions)
        accuracies.append(accuracy)

    # Find the threshold that gives the maximum accuracy
    best_index = np.argmax(accuracies)
    best_threshold = thresholds[best_index]
    best_accuracy = accuracies[best_index]

    return best_threshold, best_accuracy


def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two batches of embeddings.
    """
    return F.cosine_similarity(embedding1, embedding2, dim=1)



def evaluate_verification(model, dataloader, device):
    """
    Evaluate verification performance on a dataset with batch support.
    """
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Compute embeddings for the batch
            embedding1 = F.normalize(model(img1))
            embedding2 = F.normalize(model(img2))

            # Compute similarity scores in batch
            scores = compute_similarity(embedding1, embedding2)  # Shape: (batch_size,)

            # Store results
            all_scores.extend(scores.cpu().tolist())  # Convert to list for easy processing
            all_labels.extend(label.cpu().tolist())  # Handle batch labels correctly

    return all_scores, all_labels


def plot_roc_curve(labels, scores, save_path=None, show=False):
    """
    Plot ROC curve and compute AUC.

    Args:
        labels (array-like): Ground-truth labels (0/1).
        scores (array-like): Similarity scores or probabilities.
        save_path (str or None): If provided, saves figure to this path.
        show (bool): If True, calls plt.show() (interactive). Otherwise closes figure.

    Returns:
        float: ROC AUC value.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    if save_path is not None or show:
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    return roc_auc
