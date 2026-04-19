import os
import torch
import time
from torch.utils.data import DataLoader

from models.architectures.cnn_model import CNNModel
from datasets.fashion_mnist.fashion_mnist_loader import load_fashion_mnist
from deletion_strategies.class_deletion import class_deletion
from unlearning_algorithms.ssn_unlearning import execute_ssn_unlearning

# Import the new plotting function
from evaluation.plot_single_run import generate_run_plots

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
    # --- RUN CONFIGURATION ---
    dataset_name = "FashionMNIST"
    model_name = "RMSProp"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_class_to_delete = 7
    batch_size = 64
    
    model_load_path = "models/trained_models/fashionmnist/learning/rmsprop_model.pth" 
    model_save_path = "models/trained_models/fashionmnist/unlearning/ssn_unlearned_model_for_rmsprop.pth"

    print(f"Loading {dataset_name} data and isolating classes...")
    train_loader, test_loader = load_fashion_mnist(batch_size=batch_size)
    base_train_dataset = train_loader.dataset
    retain_dataset, target_dataset = class_deletion(base_train_dataset, class_to_delete=target_class_to_delete)
    
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    print("Initializing CNNModel for FashionMNIST...")
    model = CNNModel(input_channels=1, num_classes=10, input_size=28).to(device)
    
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f"Baseline model loaded successfully from {model_load_path}.")
    else:
        print(f"Warning: Baseline model not found at {model_load_path}.")

    print("\n--- BASELINE SANITY CHECK ---")
    base_target = evaluate_accuracy(model, target_loader, device)
    base_retain = evaluate_accuracy(model, retain_loader, device)
    print(f"Pre-Unlearning Target Class Accuracy:      {base_target:.2f}%")
    print(f"Pre-Unlearning Retained Classes Accuracy:  {base_retain:.2f}%")
    print("-----------------------------\n")

    # ---------------------------------------------------------
    # TIMER START
    # ---------------------------------------------------------
    start_time = time.time()
    print("Initiating Selective Semantic Necrosis (SSN)...")
    
    unlearned_model = execute_ssn_unlearning(
        model=model,
        target_loader=target_loader,
        retain_loader=retain_loader,
        epochs=3,
        lr=0.005
    )
    
    # ---------------------------------------------------------
    # TIMER END
    # ---------------------------------------------------------
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("SSN Unlearning Complete.\n")
    print(f"=== SSN Execution Time: {execution_time:.4f} seconds ===\n")

    print("Evaluating SSN Isolation and Destruction...")
    target_acc = evaluate_accuracy(unlearned_model, target_loader, device)
    retain_acc = evaluate_accuracy(unlearned_model, retain_loader, device)

    print(f"Target Class Accuracy:      {target_acc:.2f}% (Goal: ~10% Uniform Noise)")
    print(f"Retained Classes Accuracy:  {retain_acc:.2f}% (Goal: Intact, close to baseline)")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(unlearned_model.state_dict(), model_save_path)
    print(f"\nNecrotic model saved to {model_save_path}")

    # ---------------------------------------------------------
    # GENERATE PLOTS
    # ---------------------------------------------------------
    print("\nGenerating evaluation plots...")
    generate_run_plots(
        dataset_name=dataset_name, 
        model_name=model_name, 
        exec_time=execution_time, 
        pre_target=base_target, 
        pre_retain=base_retain, 
        post_target=target_acc, 
        post_retain=retain_acc
    )