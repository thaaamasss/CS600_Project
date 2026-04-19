import os
import torch
from torch.utils.data import DataLoader

from models.architectures.cnn_model import CNNModel
from datasets.cifar10.cifar10_loader import load_cifar10
from deletion_strategies.class_deletion import class_deletion

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
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
    
    model_path = "models/trained_models/cifar10/unlearning/finetuning_model.pth" 

    print("Loading the 9 Retained Classes...")
    train_loader, _ = load_cifar10(batch_size=64)
    retain_dataset, _ = class_deletion(train_loader.dataset, class_to_delete=target_class)
    retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=False)

    print("Loading the Fine-Tuned Model...")
    model = CNNModel(input_channels=3, num_classes=10, input_size=32).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Could not find {model_path}")
        exit()

    print("\nAuditing Collateral Damage (The Onion Effect)...")
    retain_acc = evaluate_accuracy(model, retain_loader, device)
    
    print("--------------------------------------------------")
    print(f"Fine-Tuned Retained Accuracy: {retain_acc:.2f}%")
    print(f"SSN Retained Accuracy:        79.86%")
    print("--------------------------------------------------")