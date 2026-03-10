import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def sisa_unlearning(model, remaining_dataset, test_loader, device,
                    batch_size=64, epochs=5, learning_rate=0.001):

    print("Starting SISA unlearning...")

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(
        remaining_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader, desc=f"SISA Epoch {epoch+1}", leave=False):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print("SISA retrain epoch:", epoch + 1, "Loss:", avg_loss)

    accuracy = evaluate(model, test_loader, device)

    return model, accuracy


def evaluate(model, test_loader, device):

    model = model.to(device)
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

    print("Test Accuracy after SISA unlearning:", accuracy)

    return accuracy