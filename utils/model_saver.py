import torch
import os


def save_model(model, dataset_name, stage, model_name):

    """
    Save model inside learning or unlearning directory.

    Example paths:

    models/trained_models/mnist/learning/adam_model.pth
    models/trained_models/mnist/unlearning/retraining_model.pth
    """

    base_dir = os.path.join(
        "models",
        "trained_models",
        dataset_name,
        stage
    )

    # Create directories if they do not exist
    os.makedirs(base_dir, exist_ok=True)

    model_path = os.path.join(base_dir, model_name)

    torch.save(model.state_dict(), model_path)

    print("Model saved at:", model_path)



def load_model(model, dataset_name, stage, model_name, device):

    """
    Load model from learning or unlearning directory.
    """

    model_path = os.path.join(
        "models",
        "trained_models",
        dataset_name,
        stage,
        model_name
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    print("Model loaded from:", model_path)

    return model