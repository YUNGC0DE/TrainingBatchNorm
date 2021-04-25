import os
import torch
from torch import optim, nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import ResNet


def train(model: ResNet, device: torch.device, train_loader: DataLoader
          , val_loader: DataLoader, arch: str
          , epochs: int = 160, lr: float = 0.1
          , momentum: float = 0.9, bn_only: bool = False) -> dict:
    """
    :param bn_only: BN only
    :param model: ResNet model
    :param device: device to inference
    :param train_loader: train loader
    :param val_loader: validation loader
    :param arch: ResNet<Deep>
    :param epochs: number of epochs
    :param lr: base learning rate
    :param momentum: momentum
    :return: dict with statistics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120])
    save_steps = (25, 50, 79, 119)
    stats = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    def train_():
        model.train(True)
        total_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {loss.item()}")
        epoch_loss = total_loss / len(train_loader)
        stats["train_loss"].append(epoch_loss)

    def save_():
        local_dir = os.path.join("snapshots", arch)
        os.makedirs(local_dir, exist_ok=True)
        if bn_only:
            save_name = os.path.join(local_dir, f"snapshot_{epoch + 1}_BN_.pth.tar")
        else:
            save_name = os.path.join(local_dir, f"snapshot_{epoch + 1}.pth.tar")
        torch.save(model.state_dict(), save_name)
        print(f"Model {save_name} saved")

    def eval_():
        model.train(False)
        total_loss = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            epoch_loss = total_loss / len(val_loader)
            epoch_accuracy = correct / total * 100
            stats["val_loss"].append(epoch_loss)
            stats["val_accuracy"].append(epoch_accuracy)
            print(f"Epoch {epoch + 1}: Accuracy: {epoch_accuracy}. Loss: {epoch_loss}")

    for epoch in tqdm(range(epochs)):
        train_()
        print(f"LR: {lr_scheduler.get_last_lr()}")
        lr_scheduler.step()
        eval_()
        if epoch in save_steps:
            save_()
    print("---------- TRAINING IS FINISHED ----------")
    return stats
