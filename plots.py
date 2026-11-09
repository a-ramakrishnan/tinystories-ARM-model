# Plots: training / validation loss curves

import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, eval_interval, save_path=None, show=True):
    """Plot train/val loss per evaluation. `show=True` opens a window and blocks until it is closed."""
    steps = [eval_interval * (i + 1) for i in range(len(train_losses))]
    plt.figure()
    plt.plot(steps, train_losses, 'g', label='train_loss')
    plt.plot(steps, val_losses, 'r', label='validation_loss')
    plt.xlabel(f"Training step (evaluated every {eval_interval})")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
