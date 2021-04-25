import os
from matplotlib import pyplot as plt


def draw_loss(stats_bn: dict, stats_random_f: dict, arch: str, save_path: str = "images") -> None:
    """
    :param arch: arch
    :param stats_bn: statistics dict with training only BatchNormalization
    :param stats_random_f: statistics dict with training random features
    :param save_path: path to save plot
    :return: None
    """
    save_path = os.path.join(save_path, arch)
    os.makedirs(save_path, exist_ok=True)
    x = list(range(1, len(stats_bn["train_loss"]) + 1))
    train_loss_bn = stats_bn["train_loss"]
    val_loss_bn = stats_bn["val_loss"]
    train_loss_random_f = stats_random_f["train_loss"]
    val_loss_random_f = stats_random_f["val_loss"]
    plt.title("Train and Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.plot(x, train_loss_bn, 'b')
    plt.plot(x, val_loss_bn, 'r')
    plt.plot(x, train_loss_random_f, 'b--')
    plt.plot(x, val_loss_random_f, 'r--')
    plt.legend(['Train loss BN only', 'Val loss BN only'
                , 'Train loss 2 F.P.C', 'Val loss 2 F.P.C'], loc=2)
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()


def draw_accuracy(stats_bn: dict, stats_random_f: dict, arch: str, save_path: str = "images") -> None:
    """
    :param arch: arch
    :param stats_bn: statistics dict with training only BatchNormalization
    :param stats_random_f: statistics dict with training random features
    :param save_path: path to save plot
    :return: None
    """
    save_path = os.path.join(save_path, arch)
    os.makedirs(save_path, exist_ok=True)
    x = list(range(1, len(stats_bn["train_loss"]) + 1))
    accuracy_bn = stats_bn["val_accuracy"]
    accuracy_random_f = stats_random_f["val_accuracy"]
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.plot(x, accuracy_bn, 'b')
    plt.plot(x, accuracy_random_f, 'r')
    plt.legend(['Accuracy BN only', 'Accuracy 2 F.P.C'], loc=2)
    plt.savefig(os.path.join(save_path, "accuracy.png"))
    plt.close()
