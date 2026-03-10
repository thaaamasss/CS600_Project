import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to train a model using SGD optimizer
def train_sgd(model, train_loader, test_loader, device, epochs=10, learning_rate=0.01):

    # Move model to CPU or GPU depending on availability
    model.to(device)

    # Loss function for classification problems
    # CrossEntropyLoss is commonly used for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Define SGD optimizer
    # SGD updates model weights using gradients computed during backpropagation
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):

        # Set model to training mode
        model.train()

        running_loss = 0

        # Iterate over batches of training data
        for images, labels in tqdm(train_loader):

            # Move batch data to CPU/GPU
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients from the previous step
            optimizer.zero_grad()

            # Forward pass
            # The model predicts class scores for the input images
            outputs = model(images)

            # Compute loss between predictions and true labels
            loss = criterion(outputs, labels)

            # Backpropagation
            # Computes gradients of the loss with respect to model parameters
            loss.backward()

            # Update model weights using SGD
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item()

        print("Epoch:", epoch + 1, "Training Loss:", running_loss)


    # After training, evaluate the model on the test dataset
    accuracy = evaluate(model, test_loader, device)

    return model, accuracy



# Function to evaluate model performance on test data
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

            # Get predicted class index
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            # Count correct predictions
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Test Accuracy:", accuracy)

    return accuracy