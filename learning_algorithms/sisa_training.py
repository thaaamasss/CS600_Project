import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.architectures.sisa_model import SISAEnsemble

def create_shards(dataset, num_shards):


    shard_size = len(dataset) // num_shards
    shards = []

    for i in range(num_shards):

        start = i * shard_size
        end = start + shard_size

        shards.append(
            torch.utils.data.Subset(
                dataset,
                list(range(start, end))
            )
        )

    return shards


def train_shard_model(model, train_loader, device, epochs=10, learning_rate=0.001):


    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print("Shard Training Epoch:", epoch + 1, "Loss:", running_loss)

    return model


def train_sisa(
model_class,
train_dataset,
test_loader,
device,
input_channels,
num_classes,
input_size,
num_shards=5,
batch_size=64
):


    print("Creating dataset shards...")

    shards = create_shards(train_dataset, num_shards)

    shard_models = []

    for i, shard in enumerate(shards):

        print("Training shard model:", i + 1)

        shard_loader = torch.utils.data.DataLoader(
            shard,
            batch_size=batch_size,
            shuffle=True
        )

        model = model_class(
            input_channels=input_channels,
            num_classes=num_classes,
            input_size=input_size
        )

        trained_model = train_shard_model(model, shard_loader, device)

        shard_models.append(trained_model)

    accuracy = evaluate_sisa(shard_models, test_loader, device)

    ensemble_model = SISAEnsemble(shard_models)

    return ensemble_model, accuracy


def evaluate_sisa(shard_models, test_loader, device):


    for model in shard_models:
        model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs_sum = None

            for model in shard_models:

                outputs = model(images)

                if outputs_sum is None:
                    outputs_sum = outputs
                else:
                    outputs_sum += outputs

            outputs_avg = outputs_sum / len(shard_models)

            _, predicted = torch.max(outputs_avg, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("SISA Test Accuracy:", accuracy)

    return accuracy
