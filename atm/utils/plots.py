import matplotlib.pyplot as plt


def plot_metrics(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    v2_accs,
    v4_accs,
    v10_accs,
    best_epoch_info,
    title="pos_img_text",
    save_path="pos_img_text.png",
):
    """Plot loss and accuracy curves with summary info.

    Args:
        train_losses (Sequence[float]): Training loss per epoch.
        test_losses (Sequence[float]): Test loss per epoch.
        train_accuracies (Sequence[float]): Training accuracy per epoch.
        test_accuracies (Sequence[float]): Test accuracy per epoch.
        v2_accs (Sequence[float]): 2-class accuracy per epoch.
        v4_accs (Sequence[float]): 4-class accuracy per epoch.
        v10_accs (Sequence[float]): 10-class accuracy per epoch.
        best_epoch_info (dict): Info for best epoch with keys:
            'epoch', 'train_loss', 'train_accuracy',
            'test_loss', 'test_accuracy', 'v2_acc',
            'v4_acc', 'v10_acc'.
        title (str): Main title for the figure.
        save_path (str): File path to save the figure.
    """
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label="Train Loss")
    axs[0, 0].plot(test_losses, label="Test Loss")
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label="Train Accuracy")
    axs[0, 1].plot(test_accuracies, label="Test Accuracy")
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label="2-class Accuracy")
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label="4-class Accuracy")
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label="10-class Accuracy")
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Annotation info
    info_text = (
        f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
        f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
        f"Train Accuracy: {best_epoch_info['train_acc']:.4f}\n"
        f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
        f"Test Accuracy: {best_epoch_info['test_acc']:.4f}\n"
        f"2-Class Acc: {best_epoch_info['v2_acc']:.4f}\n"
        f"4-Class Acc: {best_epoch_info['v4_acc']:.4f}\n"
        f"10-Class Acc: {best_epoch_info['v10_acc']:.4f}"
    )
    axs[2, 1].axis("off")
    axs[2, 1].text(
        0.5,
        0.5,
        info_text,
        fontsize=10,
        ha="center",
        va="center",
        transform=axs[2, 1].transAxes,
    )

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.savefig(save_path)
    plt.close(fig)
 