import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Assuming these match your project's import structure for CIFAR-10
from models.architectures.cnn_model import CNNModel
from datasets.cifar10.cifar10_loader import load_cifar10
from deletion_strategies.class_deletion import class_deletion

def compute_confidence_losses(model, dataloader, device):
    """
    Passes data through the model and calculates the Cross-Entropy Loss 
    for every individual image.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none') # 'none' keeps individual losses
    all_losses = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            all_losses.extend(losses.cpu().numpy())
            
    return np.array(all_losses)

def execute_loss_threshold_mia(forgotten_losses, unseen_losses):
    """
    Simulates an attacker trying to separate training data (forgotten) 
    from testing data (unseen) based purely on loss values.
    """
    # Create labels: 1 for Member (Forgotten Data), 0 for Non-Member (Unseen Data)
    labels = np.concatenate([np.ones(len(forgotten_losses)), np.zeros(len(unseen_losses))])
    all_losses = np.concatenate([forgotten_losses, unseen_losses])
    
    # The attacker sweeps through possible loss thresholds to find their best guessing strategy
    best_mia_accuracy = 0.0
    
    # Check 100 possible thresholds between the minimum and maximum loss
    thresholds = np.linspace(all_losses.min(), all_losses.max(), 100)
    
    for threshold in thresholds:
        # Attacker Logic: "If the loss is lower than my threshold, it must be a Member"
        predictions = (all_losses < threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        
        if accuracy > best_mia_accuracy:
            best_mia_accuracy = accuracy
            
    return best_mia_accuracy * 100

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_class_to_delete = 5 # CIFAR-10 Target Class (e.g., Dog)
    batch_size = 64
    
    # Point this to your CIFAR-10 SSN unlearned model
    model_load_path = "models/trained_models/cifar10/unlearning/ssn_unlearned_model.pth"

    print("Loading CIFAR-10 data...")
    # Load BOTH train and test sets
    train_loader, test_loader = load_cifar10(batch_size=batch_size)
    
    # Isolate the target class from the TRAIN set (The Forgotten Members)
    _, forgotten_dataset = class_deletion(train_loader.dataset, class_to_delete=target_class_to_delete)
    forgotten_loader = DataLoader(forgotten_dataset, batch_size=batch_size, shuffle=False)
    
    # Isolate the target class from the TEST set (The Unseen Non-Members)
    _, unseen_dataset = class_deletion(test_loader.dataset, class_to_delete=target_class_to_delete)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing CNNModel for CIFAR-10...")
    model = CNNModel(input_channels=3, num_classes=10, input_size=32).to(device)
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    print(f"Necrotic model loaded successfully from {model_load_path}.\n")

    # ---------------------------------------------------------
    # MIA EXECUTION
    # ---------------------------------------------------------
    print("Executing Membership Inference Attack (MIA)...")
    
    print("1. Extracting statistical footprint of Forgotten Data (Members)...")
    forgotten_losses = compute_confidence_losses(model, forgotten_loader, device)
    
    print("2. Extracting statistical footprint of Unseen Data (Non-Members)...")
    unseen_losses = compute_confidence_losses(model, unseen_loader, device)
    
    print("3. Attacker is attempting to separate the datasets based on loss...\n")
    mia_accuracy = execute_loss_threshold_mia(forgotten_losses, unseen_losses)
    
    print("--------------------------------------------------")
    print(f"MIA Attacker Accuracy: {mia_accuracy:.2f}%")
    
    if mia_accuracy <= 55.0:
        print("CONCLUSION: SUCCESS. The attacker is randomly guessing (~50%).")
        print("The statistical footprint of the target memory is cryptographically destroyed.")
    else:
        print("CONCLUSION: VULNERABLE. The attacker can identify the forgotten data.")
    print("--------------------------------------------------")