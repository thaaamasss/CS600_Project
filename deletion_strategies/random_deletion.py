import random
import torch


# Function to randomly select samples that will be deleted
def random_deletion(dataset, num_delete):

    """
    dataset: the full training dataset
    num_delete: number of samples to remove

    returns:
        remaining_dataset -> dataset without deleted samples
        deleted_dataset -> samples that were removed
    """

    # Total number of samples in the dataset
    dataset_size = len(dataset)

    # Create a list of all dataset indices
    all_indices = list(range(dataset_size))

    # Randomly choose indices to delete
    delete_indices = random.sample(all_indices, num_delete)

    # Remaining indices are those not selected for deletion
    remaining_indices = list(set(all_indices) - set(delete_indices))

    # Create subsets using the selected indices
    deleted_dataset = torch.utils.data.Subset(dataset, delete_indices)
    remaining_dataset = torch.utils.data.Subset(dataset, remaining_indices)

    print("Total samples in dataset:", dataset_size)
    print("Randomly deleted samples:", len(delete_indices))
    print("Remaining samples:", len(remaining_indices))

    return remaining_dataset, deleted_dataset