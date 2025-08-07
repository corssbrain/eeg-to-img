"""Utility for visualising training metrics in a clean 2 × 3 grid."""

from typing import Mapping, Sequence, Any

import matplotlib.pyplot as plt


def plot_metrics(
    train_losses: Sequence[float],
    test_losses: Sequence[float],
    train_accuracies: Sequence[float],
    test_accuracies: Sequence[float],
    v2_accs: Sequence[float],
    v4_accs: Sequence[float],
    v10_accs: Sequence[float],
    best_epoch_info: Mapping[str, Any], 
    save_path: str = "retrival_results.png",
) -> None:
    """
    Plot loss/accuracy curves plus a text panel (2 × 3 layout).

    The top-right and right-hand spines of every subplot are removed for a
    cleaner look.

    Args
    ----
    train_losses, test_losses
        Loss values per epoch.
    train_accuracies, test_accuracies
        Accuracy values per epoch (0–1 or 0–100 %).
    v2_accs, v4_accs, v10_accs
        Per-epoch accuracies for the 2-, 4- and 10-class tasks.
    best_epoch_info
        Dict containing at least the keys
        ``epoch``, ``train_loss``, ``train_accuracy``, ``test_loss``,
        ``test_accuracy``, ``v2_acc``, ``v4_acc`` and ``v10_acc``.
    title
        Figure-level title.
    save_path
        Where to write the figure.
    """
    # A pleasant default style
    plt.style.use("ggplot")

    fig, axes = plt.subplots(1, 6, figsize=(28, 4), dpi=120)
    axes = axes.ravel()  # flatten to 1-D for easy indexing

    # ------------------------------------------------------------------ plots
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(test_losses,  label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(train_accuracies, label="Train")
    axes[1].plot(test_accuracies,  label="Test")
    axes[1].set_title("Overall Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[2].plot(v2_accs, label="2-Class")
    axes[2].set_title("2-Class Accuracy")
    axes[2].set_xlabel("Epoch")

    axes[3].plot(v4_accs, label="4-Class")
    axes[3].set_title("4-Class Accuracy")
    axes[3].set_xlabel("Epoch")

    axes[4].plot(v10_accs, label="10-Class")
    axes[4].set_title("10-Class Accuracy")
    axes[4].set_xlabel("Epoch")

    # ---------------------------------------------------------------- annotate
    axes[5].axis("off")
    info_text = (
        f"Best Model (Epoch {best_epoch_info['epoch']})\n"
        f"Train loss  : {best_epoch_info['train_loss']:.4f}\n"
        f"Train acc  : {best_epoch_info['train_acc']:.4f}\n"
        f"Test loss  : {best_epoch_info['test_loss']:.4f}\n"
        f"Test acc  : {best_epoch_info['test_acc']:.4f}\n"
        f"2-Class acc : {best_epoch_info['v2_acc']:.4f}\n"
        f"4-Class acc : {best_epoch_info['v4_acc']:.4f}\n"
        f"10-Class acc: {best_epoch_info['v10_acc']:.4f}"
    )
 
    axes[5].text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=20,
        transform=axes[5].transAxes,
    )

    # ----------------------------------------------------------- formatting
    for ax in axes[:5]:  # skip the annotation panel
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        # remove top & right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)  # leave room for the suptitle
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

 