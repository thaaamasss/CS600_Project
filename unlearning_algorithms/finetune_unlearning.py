import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to perform unlearning using fine-tuning
def finetune_unlearning(model, remaining_dataset, test_loader, device,
                        batch_size=64, epochs=5, learning_rate=0.0005):

    """
    model: previously trained model
    remaining_dataset: dataset after deletion
    test_loader: dataset used to evaluate the model
    device: cpu or cuda
    """

    print("Starting fine-tuning for unlearning...")

    # Move model to device
    model = model.to(device)

    # DataLoader for remaining dataset
    train_loader = torch.utils.data.DataLoader(
        remaining_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (smaller LR for fine-tuning)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"FineTune Epoch {epoch+1}"):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print("Fine-tune Epoch:", epoch + 1, "Training Loss:", running_loss)

    # Evaluate model after fine-tuning
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

    print("Test Accuracy after fine-tuning:", accuracy)

    return accuracy