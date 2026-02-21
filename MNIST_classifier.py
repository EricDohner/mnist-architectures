import numpy as np, sys, torch, argparse, torchvision, matplotlib.pyplot as plt
from torch import nn
import networks, viz, transformer
from sklearn.model_selection import train_test_split

def main():
    # parser is not going to be implemented right now since VSCode makes you jump through absurd hoops to run with command-line inputs
    # parser = argparse.ArgumentParser(prog="MNIST_sigma" ,description="MNIST classifier")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    print(f"---- Using {device}")
    #following https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html to learn torch syntax. However, we're just doing regular MNIST here rather than FashionMNIST.
    #All that this really changes is number of outputs.
    train_data = torchvision.datasets.MNIST(root="data", train=True, download=True, transform = 
                                            torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    test_data = torchvision.datasets.MNIST(root="data", train=False, download=True, transform = 
                                           torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(), 
                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    print("---- Data imported")

    batch_size = 50
    epochs = 5
    val_size = 5000
    interval = 100

    learning_rate = 0.03

    # model = networks.CNN_res()
    model = transformer.mnist_transformer(128, 8, 28, 7, 4)
    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.001, weight_decay=0.001, lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_data)),
        train_data.targets,
        stratify = train_data.targets,
        test_size = val_size
    )


    train_split = torch.utils.data.Subset(train_data, train_indices)
    val_split = torch.utils.data.Subset(train_data, val_indices)

    print("---- Test and validation data split")

    train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_split, batch_size = batch_size*10, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size*10, shuffle=False)

    print("---- Train, validation, and test dataloaders created")

    # model = networks.MLP(28*28,128,3,10,nn.LeakyReLU)
    # model = torch.compile(model)
    model = model.to(device)

    print("---- Model, loss function, and optimizer created")

    model.train()

    # layers, grads = viz.get_all_layers(model, viz.hook_forward, viz.hook_backward)

    train_loss_list, val_loss_list, acc_list, batch_list = networks.train_model(
        model, epochs, train_dataloader, val_dataloader, learning_rate, lossfn, optimizer, scheduler, device, interval
        )

    model.eval()

    val_size = len(val_dataloader.dataset)

    correct, test_loss = networks.test_model(model, test_dataloader, lossfn, device)
    acc = 100*correct

    train_loss = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            train_loss += lossfn(pred, y).item()

    train_loss /= len(train_dataloader)

    print(f'-- Accuracy on TESTING set = {(acc):>0.2f}% -- train loss, test loss = {(train_loss):>0.3f}, {(test_loss):>0.3f}')

    # layer_idx, avg_grads = viz.get_grads(grads)
    # avg_grads = [x.to("cpu") for x in avg_grads]

    params, grad_list = viz.get_grad_norms(model)

    viz.plot_training_metrics(batch_list, train_loss_list, val_loss_list, acc_list, params, grad_list)


if __name__ == '__main__':
    main()