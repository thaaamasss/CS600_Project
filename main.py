import argparse


# Import experiment functions
from experiments.mnist_experiment import run_mnist_experiment
from experiments.fashion_mnist_experiment import run_fashion_mnist_experiment
from experiments.cifar10_experiment import run_cifar10_experiment
from experiments.cifar100_experiment import run_cifar100_experiment


def main():

    # Argument parser allows selecting dataset from command line
    parser = argparse.ArgumentParser(description="Machine Unlearning Experiment Runner")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to run experiment on: mnist, fashion_mnist, cifar10, cifar100"
    )

    args = parser.parse_args()

    dataset = args.dataset.lower()

    print("Selected dataset:", dataset)

    # Run experiment based on selected dataset
    if dataset == "mnist":
        run_mnist_experiment()

    elif dataset == "fashion_mnist":
        run_fashion_mnist_experiment()

    elif dataset == "cifar10":
        run_cifar10_experiment()

    elif dataset == "cifar100":
        run_cifar100_experiment()

    else:
        raise ValueError("Unsupported dataset: " + dataset)


if __name__ == "__main__":
    main()