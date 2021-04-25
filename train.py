import torch
import argparse
from scripts.preprocessing import get_frozen_models, get_loader
from scripts.processing import train
from scripts.postprocessing import draw_loss, draw_accuracy
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet')
    parser.add_argument('-D', required=True, type=int, help='ResNet deep [32, 56, 101]')
    return parser.parse_args()


def load_model(res_deep: int) -> ResNet:
    """
    :param res_deep: ResNet deep
    :return: ResNet model
    """
    arch_dict = {
        32: resnet32,
        56: resnet56,
        110: resnet110,
    }
    if arch_dict.get(res_deep) is not None:
        return arch_dict[res_deep]()
    raise AttributeError(f"No such deep {res_deep} in ResNet")


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.D)
    model_bn, model_random_f = get_frozen_models(model)
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = get_loader()

    print(f"Model ResNet{args.D} is ready to train on {device}")
    arch = f"ResNet{args.D}"
    print("---------- RANDOM FEATURES TRAINING ----------")
    model_random_f.to(device)
    stats_random_f = train(model_random_f, device, train_loader, val_loader, arch)
    print("---------- BN ONLY TRAINING ----------")
    model_bn.to(device)
    stats_bn = train(model_bn, device, train_loader, val_loader, arch, bn_only=True)
    draw_loss(stats_bn, stats_random_f, arch)
    draw_accuracy(stats_bn, stats_random_f, arch)
