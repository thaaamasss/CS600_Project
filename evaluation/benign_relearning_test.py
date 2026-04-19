import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models.architectures.cnn_model import CNNModel
from datasets.fashion_mnist.fashion_mnist_loader import load_fashion_mnist
from deletion_strategies.class_deletion import class_deletion

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_class = 5  
    model_path = "models/trained_models/fashionmnist/unlearning/ssn_unlearned_model_for_rmsprop.pth"

    print("Loading data for the Stress Test...")
    train_loader, _ = load_fashion_mnist(batch_size=64)
    _, target_dataset = class_deletion(train_loader.dataset, class_to_delete=target_class)

    # Extract a micro-batch of just 50 images of the deleted class
    micro_batch_indices = list(range(50))
    micro_batch_dataset = Subset(target_dataset, micro_batch_indices)
    
    micro_loader = DataLoader(micro_batch_dataset, batch_size=50, shuffle=True)
    eval_loader = DataLoader(target_dataset, batch_size=64, shuffle=False)

    print("Loading the Necrotic Model...")
    model = CNNModel(input_channels=1, num_classes=10, input_size=28).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Could not find the SSN model at {model_path}")
        exit()

    pre_test_acc = evaluate_accuracy(model, eval_loader, device)
    print(f"\nPre-Stress Test Target Accuracy: {pre_test_acc:.2f}% (Should be ~0.00%)")

    print("\nInitiating Benign Relearning (The Stress Test)...")
    model.train()
    # We use Adam with a standard learning rate to give the model the best possible chance to relearn
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()

    # Train for exactly 2 epochs on the tiny micro-batch
    epochs = 2
    for epoch in range(epochs):
        for inputs, labels in micro_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{epochs} complete.")

    post_test_acc = evaluate_accuracy(model, eval_loader, device)
    print(f"\nPost-Stress Test Target Accuracy: {post_test_acc:.2f}%")
    print("--------------------------------------------------")
    
    if post_test_acc < 25.0:
        print("CONCLUSION: SUCCESS. The memory is permanently destroyed.")
        print("The model cannot rapidly recover the data via the Relearning Vulnerability.")
    else:
        print("CONCLUSION: FAILURE. The model rapidly recovered the memory.")
        print("The necrosis was superficial.")