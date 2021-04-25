import torch
import argparse
from scripts.preprocessing import load_model, get_frozen_models, get_loader
from scripts.processing import train
from scripts.postprocessing import draw_loss, draw_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet')
    parser.add_argument('-D', required=True, type=int, help='ResNet deep [18, 34, 50]')
    parser.add_argument('-G', default=0, type=int, help='Gpu id')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.D)
    model_bn, model_random_f = get_frozen_models(model)
    device = torch.device(f"cuda:{args.G}" if torch.cuda.is_available() else 'cpu')
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
