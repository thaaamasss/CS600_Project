import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to retrain the model from scratch after deletion
def retraining_unlearning(
        model_class,
        remaining_dataset,
        test_loader,
        device,
        input_channels,
        num_classes,
        input_size,
        batch_size=64,
        epochs=10,
        learning_rate=0.001):

    """
    model_class: model architecture (e.g., CNNModel)
    remaining_dataset: dataset after deletion
    test_loader: dataset used for evaluation
    device: cpu or cuda
    """

    print("Starting full retraining for unlearning...")

    # Create a fresh model instance
    model = model_class(
        input_channels=input_channels,
        num_classes=num_classes,
        input_size=input_size
    )

    # Move model to device
    model.to(device)

    # DataLoader for the remaining dataset
    train_loader = torch.utils.data.DataLoader(
        remaining_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print("Epoch:", epoch + 1, "Training Loss:", running_loss)

    # Evaluate retrained model
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

    print("Test Accuracy after retraining:", accuracy)

    return accuracy