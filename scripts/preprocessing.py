import copy
import random
from random import randint
from torchvision import models, transforms, datasets
from typing import Tuple
from prettytable import PrettyTable
import torch
from torch.utils.data import DataLoader


def get_loader(batch_size: int = 128, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    :param batch_size: batch size
    :param num_workers: num workers
    :return: loaders
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def get_frozen_models(model: models.ResNet) -> Tuple[models.ResNet, models.ResNet]:
    """
    :param model: ResNet model
    :return: table
    """
    def model_bn_():
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        bn_params = 0
        for name, parameter in model.named_parameters():
            params = parameter.numel()
            if "bn" not in name:
                parameter.requires_grad_(False)
                table.add_row([name, f"frozen({params})"])
            else:
                bn_params += params
                table.add_row([name, params])
            total_params += params
        print("------------------- BN ONLY -------------------")
        print(table)
        print(f"Total Params: {total_params}")
        print(f"BN Trainable Params: {bn_params}")
        print(f"Left {round(bn_params / total_params * 100, 3)}% params \n")

    def model_random_f_():
        model_random_f.requires_grad_(False)
        for m in model_random_f.modules():
            if isinstance(m, torch.nn.Conv2d):
                weights = m.state_dict()
                filter_size = len(weights['weight'][0][0][0])
                random_x_1 = randint(0, filter_size - 1)
                random_y_1 = randint(0, filter_size - 1)
                random_x_2 = randint(0, filter_size - 1)
                random_y_2 = randint(0, filter_size - 1)
                channel_1 = randint(0, len(weights['weight'][0]) - 1)
                channel_2 = randint(0, len(weights['weight'][0]) - 1)
                weights['weight'] = torch.zeros_like(weights['weight'])
                weights['weight'][channel_1, :, random_x_1, random_y_1] = random.uniform(0.001, 1)
                weights['weight'][channel_2, :, random_x_2, random_y_2] = random.uniform(0.001, 1)
                m.load_state_dict(weights)
                m.requires_grad_(True)
        model.conv1.requires_grad_(False)

    model_random_f = copy.deepcopy(model)
    model_bn_()
    model_random_f_()
    return model, model_random_f
