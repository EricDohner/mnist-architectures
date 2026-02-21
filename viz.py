import numpy as np, sys, torch, torchvision, matplotlib.pyplot as plt
from torch import nn

def get_grad_norms(model):
    num_params = 0
    grad_list = []
    params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            num_params +=1
            params.append(num_params)
            grad_list.append(param.grad.norm().item())
            if torch.isnan(param.grad).any():
                print("NaN gradient detected in parameter: " + name)
    return params, grad_list

def plot_training_metrics(batch_list, train_loss_list, val_loss_list, acc_list, layer_idx, avg_grads):
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))

    # First subplot: losses and accuracy
    ax1.plot(batch_list, train_loss_list, label='Train Loss', color='blue')
    ax1.plot(batch_list, val_loss_list, label='Val Loss', color='orange')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    # Second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(batch_list, acc_list, label='Accuracy', color='green')
    ax2.set_ylabel('Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    # Second subplot: gradient flow
    ax3.plot(layer_idx, avg_grads)
    # ax3.semilogy()
    ax3.set_xlabel("Layer depth")
    ax3.set_ylabel("Gradient norm")
    ax3.set_title("Gradient flow")
    ax3.grid(True)
    # ax3.legend()

    plt.tight_layout()
    plt.show()