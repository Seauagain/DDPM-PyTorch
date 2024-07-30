"""

"""
import numpy as np 
import os 
import matplotlib 
import matplotlib.pyplot as plt 
import math


def saveLoss(train_loss, valid_loss, args):
    r"""Save training loss and validation loss and draw the loss curves    
    """
    loss_path = os.path.join(args.model_path, 'loss')
    # os.makedirs(loss_path,exist_ok=True)
    train_file = os.path.join(loss_path, 'train_loss.npy')
    valid_file = os.path.join(loss_path, 'valid_loss.npy')
    np.save(train_file, train_loss)
    np.save(valid_file, valid_loss)
    plotLoss(args)

def plotLoss(args, axis='semilogy', dpi=200):
    r"""Draw and save training loss and validation loss curves.
    """
    loss_path = os.path.join(args.model_path, 'loss')
    losspic_path = os.path.join(loss_path, f'{args.model_name}_loss.png')
    train_file = os.path.join(loss_path, 'train_loss.npy')
    valid_file = os.path.join(loss_path, 'valid_loss.npy')
    train_loss = np.load(train_file)
    valid_loss = np.load(valid_file)

    n_iters = len(train_loss)

    plot_methods = {'semilogy': plt.semilogy, 'loglog': plt.loglog}
    #funciton handle
    plot_handle = plot_methods[axis]
    p1, = plot_handle(range(1, n_iters + 1),
                        train_loss,
                        color="chocolate",
                        linewidth=2,
                        alpha=1)
    p2, = plot_handle(range(1, n_iters + 1),
                        valid_loss,
                        color="forestgreen",
                        linestyle='-.',
                        linewidth=2,
                        alpha=1)

    order = math.floor(np.min(np.log10(train_loss)))
    plt.ylim(10**order, 1)
    plt.xlabel("epoch")
    plt.ylabel(f'loss')
    # plt.yticks([1e-3, 1e-2, 1e-1, 1e0])
    plt.legend([p1, p2], ["training loss", 'validation loss'], loc="upper right")
    plt.savefig(losspic_path, dpi=dpi)
    plt.close()