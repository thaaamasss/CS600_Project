import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to train the model using the Adam optimizer
def train_adam(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001):

    # Move the model to the selected device (CPU or GPU)
    model.to(device)

    # CrossEntropyLoss is used for multi-class classification problems
    criterion = nn.CrossEntropyLoss()

    # Define Adam optimizer
    # Adam adapts the learning rate automatically during training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loop over the dataset multiple times (epochs)
    for epoch in range(epochs):

        # Set model to training mode
        model.train()

        running_loss = 0

        # Iterate through batches of training data
        for images, labels in tqdm(train_loader):

            # Move batch to device
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(images)

            # Compute the loss between predictions and actual labels
            loss = criterion(outputs, labels)

            # Backpropagation: compute gradients
            loss.backward()

            # Update model parameters using Adam optimizer
            optimizer.step()

            # Accumulate loss for reporting
            running_loss += loss.item()

        print("Epoch:", epoch + 1, "Training Loss:", running_loss)


    # After training is complete, evaluate model on test data
    accuracy = evaluate(model, test_loader, device)

    return model, accuracy



# Function to evaluate the trained model
def evaluate(model, test_loader, device):

    # Set model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation during evaluation
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class (highest score)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            # Count correct predictions
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Test Accuracy:", accuracy)

    return accuracy