import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to approximate influence-based unlearning
def influence_unlearning(model, deleted_dataset, test_loader, device,
                         batch_size=64, learning_rate=0.0005):

    """
    model: trained model
    deleted_dataset: samples that must be forgotten
    test_loader: dataset used for evaluation
    device: cpu or cuda
    """

    print("Starting influence-based unlearning...")

    # Move model to device
    model = model.to(device)

    # DataLoader for deleted samples
    delete_loader = torch.utils.data.DataLoader(
        deleted_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training mode
    model.train()

    # Reverse gradient contribution of deleted samples
    for images, labels in tqdm(delete_loader, desc="Influence Unlearning"):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        # Compute gradients
        loss.backward()

        # Reverse gradient direction
        for param in model.parameters():
            if param.grad is not None:
                param.grad.neg_()

        # Apply reversed gradient update
        optimizer.step()

    # Evaluate model after influence removal
    accuracy = evaluate(model, test_loader, device)

    return model, accuracy


# Function to evaluate model performance
def evaluate(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Test Accuracy after influence unlearning:", accuracy)

    return accuracy