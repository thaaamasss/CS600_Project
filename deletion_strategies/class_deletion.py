import torch


# Function to delete all samples belonging to a specific class
def class_deletion(dataset, class_to_delete):

    """
    dataset: full training dataset
    class_to_delete: label that must be removed

    returns:
        remaining_dataset -> dataset without that class
        deleted_dataset -> samples belonging to that class
    """

    delete_indices = []
    remaining_indices = []

    # Iterate through the dataset
    for i in range(len(dataset)):

        # Each dataset item contains (image, label)
        _, label = dataset[i]

        # If the label matches the class to delete
        if label == class_to_delete:
            delete_indices.append(i)
        else:
            remaining_indices.append(i)

    # Create dataset subsets
    deleted_dataset = torch.utils.data.Subset(dataset, delete_indices)
    remaining_dataset = torch.utils.data.Subset(dataset, remaining_indices)

    print("Class selected for deletion:", class_to_delete)
    print("Samples deleted:", len(delete_indices))
    print("Remaining samples:", len(remaining_indices))

    return remaining_dataset, deleted_dataset