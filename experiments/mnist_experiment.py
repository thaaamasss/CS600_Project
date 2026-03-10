import torch
import time
import copy

from utils.dataset_loader import load_dataset
from utils.model_saver import save_model
from models.architectures.cnn_model import CNNModel

from learning_algorithms.sgd_training import train_sgd
from learning_algorithms.adam_training import train_adam
from learning_algorithms.rmsprop_training import train_rmsprop
from learning_algorithms.sisa_training import train_sisa

from deletion_strategies.random_deletion import random_deletion

from unlearning_algorithms.retraining_unlearning import retraining_unlearning
from unlearning_algorithms.finetune_unlearning import finetune_unlearning
from unlearning_algorithms.influence_unlearning import influence_unlearning
from unlearning_algorithms.sisa_unlearning import sisa_unlearning
from evaluation.metrics import evaluate_model
from evaluation.evaluate_learning import evaluate_learning_algorithms
from evaluation.evaluate_unlearning import evaluate_unlearning_algorithms

from utils.config import DELETE_SAMPLES

def run_mnist_experiment():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running MNIST experiment on:", device)

    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------

    train_loader, test_loader = load_dataset("mnist")
    train_dataset = train_loader.dataset

    # ------------------------------------------------
    # TRAIN USING SGD
    # ------------------------------------------------

    model_sgd = CNNModel(input_channels=1, num_classes=10, input_size=28)

    start = time.time()

    trained_sgd, acc_sgd = train_sgd(
        model_sgd,
        train_loader,
        test_loader,
        device
    )

    sgd_time = time.time() - start
    save_model(trained_sgd, "mnist", "learning", "sgd_model.pth")

    # ------------------------------------------------
    # TRAIN USING ADAM
    # ------------------------------------------------

    model_adam = CNNModel(input_channels=1, num_classes=10, input_size=28)

    start = time.time()

    trained_adam, acc_adam = train_adam(
        model_adam,
        train_loader,
        test_loader,
        device
    )

    adam_time = time.time() - start
    save_model(trained_adam, "mnist", "learning", "adam_model.pth")

    # ------------------------------------------------
    # TRAIN USING RMSPROP
    # ------------------------------------------------

    model_rms = CNNModel(input_channels=1, num_classes=10, input_size=28)

    start = time.time()

    trained_rms, acc_rms = train_rmsprop(
        model_rms,
        train_loader,
        test_loader,
        device
    )

    rms_time = time.time() - start
    save_model(trained_rms, "mnist", "learning", "rmsprop_model.pth")

    # ------------------------------------------------
    # TRAIN USING SISA
    # ------------------------------------------------

    # model_sisa = CNNModel(input_channels=1, num_classes=10, input_size=28)

    start = time.time()

    trained_sisa, acc_sisa = train_sisa(
        CNNModel,
        train_loader.dataset,
        test_loader,
        device,
        input_channels=1,
        num_classes=10,
        input_size=28
    )

    sisa_time = time.time() - start

    save_model(trained_sisa, "mnist", "learning", "sisa_model.pth")

    # ------------------------------------------------
    # STORE LEARNING RESULTS
    # ------------------------------------------------

    learning_results = {
        "SGD": {"accuracy": acc_sgd, "time": sgd_time, "loss": 0.0},
        "Adam": {"accuracy": acc_adam, "time": adam_time, "loss": 0.0},
        "RMSProp": {"accuracy": acc_rms, "time": rms_time, "loss": 0.0},
        "SISA": {"accuracy": acc_sisa, "time": sisa_time, "loss": 0.0}
    }

    evaluate_learning_algorithms(learning_results, "mnist")

    # ------------------------------------------------
    # SELECT BEST LEARNING ALGORITHM
    # Accuracy first, time as tie-breaker
    # ------------------------------------------------

    best_algorithm = sorted(
        learning_results.items(),
        key=lambda x: (-x[1]["accuracy"], x[1]["time"])
    )[0][0]

    print("Best learning algorithm:", best_algorithm)

    trained_models = {
        "SGD": trained_sgd,
        "Adam": trained_adam,
        "RMSProp": trained_rms,
        "SISA": trained_sisa
    }

    best_model = trained_models[best_algorithm]

    # ------------------------------------------------
    # APPLY DELETION STRATEGY
    # ------------------------------------------------

    remaining_dataset, deleted_dataset = random_deletion(
        train_dataset,
        DELETE_SAMPLES
    )

    # ------------------------------------------------
    # UNLEARNING : RETRAINING
    # ------------------------------------------------

    start = time.time()

    retrained_model, retrain_acc = retraining_unlearning(
        CNNModel,
        remaining_dataset,
        test_loader,
        device,
        input_channels=1,
        num_classes=10,
        input_size=28
    )

    retrain_time = time.time() - start
    deleted_acc_retrain = evaluate_model(
        retrained_model,
        deleted_dataset,
        device
    )
    save_model(retrained_model, "mnist", "unlearning", "retraining_model.pth")

    # ------------------------------------------------
    # UNLEARNING : FINETUNING
    # ------------------------------------------------

    model_copy = copy.deepcopy(best_model)

    start = time.time()

    finetuned_model, finetune_acc = finetune_unlearning(
        model_copy,
        remaining_dataset,
        test_loader,
        device
    )

    finetune_time = time.time() - start
    deleted_acc_finetune = evaluate_model(
        finetuned_model,
        deleted_dataset,
        device
    )
    save_model(finetuned_model, "mnist", "unlearning", "finetune_model.pth")

    # ------------------------------------------------
    # UNLEARNING : INFLUENCE METHOD
    # ------------------------------------------------

    model_copy = copy.deepcopy(best_model)

    start = time.time()

    influence_model, influence_acc = influence_unlearning(
        model_copy,
        deleted_dataset,
        test_loader,
        device
    )

    influence_time = time.time() - start
    deleted_acc_influence = evaluate_model(
        influence_model,
        deleted_dataset,
        device
    )
    save_model(influence_model, "mnist", "unlearning", "influence_model.pth")

    # ------------------------------------------------
    # UNLEARNING : SISA
    # ------------------------------------------------

    model_copy = copy.deepcopy(best_model)

    start = time.time()

    sisa_unlearn_model, sisa_unlearn_acc = sisa_unlearning(
        model_copy,
        remaining_dataset,
        test_loader,
        device
    )

    sisa_unlearn_time = time.time() - start
    deleted_acc_sisa = evaluate_model(
        sisa_unlearn_model,
        deleted_dataset,
        device
    )
    save_model(sisa_unlearn_model, "mnist", "unlearning", "sisa_unlearn_model.pth")

    # ------------------------------------------------
    # STORE UNLEARNING RESULTS
    # ------------------------------------------------

    unlearning_results = {
        "Retraining": {
            "remaining_accuracy": retrain_acc,
            "deleted_accuracy": deleted_acc_retrain,
            "time": retrain_time
        },
        "FineTuning": {
            "remaining_accuracy": finetune_acc,
            "deleted_accuracy": deleted_acc_finetune,
            "time": finetune_time
        },
        "Influence": {
            "remaining_accuracy": influence_acc,
            "deleted_accuracy": deleted_acc_influence,
            "time": influence_time
        },
        "SISA": {
            "remaining_accuracy": sisa_unlearn_acc,
            "deleted_accuracy": deleted_acc_sisa,
            "time": sisa_unlearn_time
        }
    }

    evaluate_unlearning_algorithms(unlearning_results, "mnist")

if __name__ == "__main__":
    run_mnist_experiment()

