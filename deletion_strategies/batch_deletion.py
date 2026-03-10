import random
import torch


# Function to delete a percentage of the dataset
def batch_deletion(dataset, deletion_percentage):

    """
    dataset: full training dataset
    deletion_percentage: percentage of samples to delete (0–100)

    returns:
        remaining_dataset -> dataset after deletion
        deleted_dataset -> samples removed in the batch
    """

    dataset_size = len(dataset)

    # Calculate number of samples to delete
    num_delete = int((deletion_percentage / 100) * dataset_size)

    # Create list of all dataset indices
    all_indices = list(range(dataset_size))

    # Randomly choose indices to delete
    delete_indices = random.sample(all_indices, num_delete)

    # Remaining indices
    remaining_indices = list(set(all_indices) - set(delete_indices))

    # Create subsets
    deleted_dataset = torch.utils.data.Subset(dataset, delete_indices)
    remaining_dataset = torch.utils.data.Subset(dataset, remaining_indices)

    print("Total dataset size:", dataset_size)
    print("Deletion percentage:", deletion_percentage, "%")
    print("Samples deleted:", len(delete_indices))
    print("Remaining samples:", len(remaining_indices))

    return remaining_dataset, deleted_dataset