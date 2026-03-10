import numpy as np
import torch

# ---------------------------------------------------
# BASIC METRIC FUNCTIONS
# ---------------------------------------------------

# Calculate classification accuracy
def compute_accuracy(correct_predictions, total_predictions):

    """
    correct_predictions: number of correct predictions
    total_predictions: total predictions made
    """

    if total_predictions == 0:
        return 0

    accuracy = correct_predictions / total_predictions

    return accuracy



# Normalize values so they fall between 0 and 1
# Used for comparing metrics with different scales
def normalize(values):

    """
    values: list of metric values
    """

    values = np.array(values)

    min_val = np.min(values)
    max_val = np.max(values)

    # Prevent division by zero
    if max_val == min_val:
        return np.ones(len(values))

    normalized = (values - min_val) / (max_val - min_val)

    return normalized



# ---------------------------------------------------
# LEARNING ALGORITHM SCORE
# ---------------------------------------------------

def compute_learning_scores(accuracy_list, time_list, loss_list):

    """
    accuracy_list: accuracy of each learning algorithm
    time_list: training time for each algorithm
    loss_list: final training loss
    """

    # Normalize metrics
    norm_accuracy = normalize(accuracy_list)

    # For time and loss, smaller values are better
    norm_time = 1 - normalize(time_list)
    norm_loss = 1 - normalize(loss_list)

    scores = []

    for i in range(len(norm_accuracy)):

        score = (
            0.6 * norm_accuracy[i] +
            0.25 * norm_time[i] +
            0.15 * norm_loss[i]
        )

        scores.append(score)

    return scores



# ---------------------------------------------------
# UNLEARNING ALGORITHM SCORE
# ---------------------------------------------------

def evaluate_model(model, dataset, device):

    from torch.utils.data import DataLoader

    model.eval()

    loader = DataLoader(dataset, batch_size=64)

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def compute_unlearning_scores(remaining_accuracy, deleted_accuracy, time_list):

    """
    remaining_accuracy: accuracy on remaining dataset
    deleted_accuracy: accuracy on deleted samples
    time_list: unlearning time
    """

    norm_remaining = normalize(remaining_accuracy)

    # Lower deleted accuracy is better (means better forgetting)
    norm_deleted = 1 - normalize(deleted_accuracy)

    norm_time = 1 - normalize(time_list)

    scores = []

    for i in range(len(norm_remaining)):

        score = (
            0.4 * norm_remaining[i] +
            0.35 * norm_deleted[i] +
            0.25 * norm_time[i]
        )

        scores.append(score)

    return scores



# ---------------------------------------------------
# FIND BEST ALGORITHM
# ---------------------------------------------------

def find_best_algorithm(algorithm_names, scores):

    """
    algorithm_names: list of algorithm names
    scores: computed scores
    """

    best_index = np.argmax(scores)

    best_algorithm = algorithm_names[best_index]

    best_score = scores[best_index]

    return best_algorithm, best_score