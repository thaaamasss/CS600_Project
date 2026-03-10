import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Function to train the model using the RMSProp optimizer
def train_rmsprop(model, train_loader, test_loader, device, epochs=10, learning_rate=0.001):

    # Move the model to CPU or GPU depending on availability
    model.to(device)

    # CrossEntropyLoss is commonly used for classification problems
    criterion = nn.CrossEntropyLoss()

    # Define RMSProp optimizer
    # RMSProp adapts learning rates by dividing the gradient
    # by a running average of its recent magnitude
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Training loop over multiple epochs
    for epoch in range(epochs):

        # Set the model to training mode
        model.train()

        running_loss = 0

        # Iterate through the training dataset batch by batch
        for images, labels in tqdm(train_loader):

            # Move images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass: compute predictions
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backpropagation: compute gradients
            loss.backward()

            # Update model parameters using RMSProp
            optimizer.step()

            # Track total loss for this epoch
            running_loss += loss.item()

        print("Epoch:", epoch + 1, "Training Loss:", running_loss)


    # After training, evaluate model performance
    accuracy = evaluate(model, test_loader, device)

    return model, accuracy



# Function to evaluate model accuracy on the test dataset
def evaluate(model, test_loader, device):

    # Set model to evaluation mode
    model.eval()

    correct = 0
    total = 0

    # Disable gradient calculations during evaluation
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