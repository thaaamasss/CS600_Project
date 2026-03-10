import torch


# Function to delete specific samples using their dataset indices
def targeted_deletion(dataset, indices_to_delete):

    """
    dataset: full training dataset
    indices_to_delete: list of dataset indices to remove

    returns:
        remaining_dataset -> dataset after removing selected samples
        deleted_dataset -> the targeted samples that were removed
    """

    dataset_size = len(dataset)

    # Convert indices_to_delete to a set for faster lookup
    delete_set = set(indices_to_delete)

    remaining_indices = []
    deleted_indices = []

    # Iterate through dataset indices
    for i in range(dataset_size):

        if i in delete_set:
            deleted_indices.append(i)
        else:
            remaining_indices.append(i)

    # Create dataset subsets
    deleted_dataset = torch.utils.data.Subset(dataset, deleted_indices)
    remaining_dataset = torch.utils.data.Subset(dataset, remaining_indices)

    print("Total dataset size:", dataset_size)
    print("Targeted samples deleted:", len(deleted_indices))
    print("Remaining samples:", len(remaining_indices))

    return remaining_dataset, deleted_dataset