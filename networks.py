import numpy as np, sys, torch, torchvision, matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

class FC_layer(nn.Module):
    def __init__(self, inputs, outputs, actfn=nn.LeakyReLU):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(inputs, outputs),
            actfn()
        )
    def forward(self, x):
        return self.stack(x)

class convpool(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size=2, pool_kernel=2, actfn=nn.LeakyReLU, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LazyConv2d(n_filters,kernel_size=kernel_size, padding=padding),
            nn.LazyBatchNorm2d(n_filters),
            nn.MaxPool2d(kernel_size=pool_kernel),
            actfn()
        )
    def forward(self, x):
        return self.layer(x)
    
class res_block(nn.Module):
    def __init__(self, n_chan, useconv = False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(n_chan, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(n_chan, kernel_size=3, padding=1, stride=strides)

        if useconv:
            self.conv3 = nn.LazyConv2d(n_chan, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)

class MLP(nn.Module):
    def __init__(self, input_features, intermediate_features, intermediate_layers, output_features, activation_fn): 
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            FC_layer(input_features, intermediate_features, activation_fn)
                            )
        for i in range(intermediate_layers):
            self.linear_stack.append(FC_layer(intermediate_features, intermediate_features, activation_fn))
        self.linear_stack.append(nn.Linear(intermediate_features, output_features))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            convpool(1, 20, kernel_size=5, pool_kernel=2, padding=2), #in: 28*28    out:14*14*20
            convpool(20,20, kernel_size=5, pool_kernel=2, padding=2), #in: 14*14*20 out: 7*7*10
            # convpool(20,10, kernel_size=2, pool_kernel=7, padding=1), #in: 7*7*20 out: 1*1*10
            self.flatten,
            FC_layer(7*7*20,100),
            # nn.Dropout(0.5),
            FC_layer(100,10)
        )

    def forward(self, x):
        return self.flatten(self.stack(x))
    
class CNN_res(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            res_block(20), #in: 28x28 ; out: 28*28*20
            res_block(20),
            res_block(20),
            nn.AvgPool2d(kernel_size=2),# out: 14*14*20
            self.flatten,
            FC_layer(14*14*20,10)
        )

    def forward(self, x):
        return self.flatten(self.stack(x))


def train_model(model, epochs, train_loader, val_loader, lr, lossfn, optimizer, scheduler, device, interval):
    #set up lists to which to append metrics
    train_loss_list = []
    validation_loss_list = []
    acc_list = []
    batch_list = []
    num_batches = len(train_loader)

    #loop over epochs
    for epoch in range(epochs):
        model.train()
        print('---- Epoch ' + str(epoch+1) + '/' + str(epochs) + ' -----' + ' Learning rate ' + str(lr) + ' ---' )

        for batch, (X, y) in enumerate(train_loader):
            
            X, y = X.to(device), y.to(device)
            loss = single_training_pass(X, y, model, optimizer, lossfn)

            if batch % interval == 0:
                correct, val_loss = test_model(model, val_loader, lossfn, device)
                acc = 100*correct
                train_loss = loss.item()
                train_loss_list.append(train_loss)
                validation_loss_list.append(val_loss)
                acc_list.append(acc)
                batch_list.append(batch + epoch*num_batches)
                print(f'-- Accuracy on validation set= {(acc):>0.2f}% -- train loss, validation loss = {(loss.item()):>0.4f}, {(val_loss):>0.4f}')
        scheduler.step()
        lr = scheduler.get_last_lr()
    return train_loss_list, validation_loss_list, acc_list, batch_list

def single_training_pass(X, y, model, optimizer, lossfn):
    optimizer.zero_grad()
    #forward pass
    output = model(X)
    #compute loss
    loss = lossfn(output, y)
    #backward pass
    loss.backward()
    #step!
    optimizer.step()
    return loss

def test_model(model, test_loader, lossfn, device):
    model.eval()
    test_loss = 0
    correct = 0
    numbatches = len(test_loader)
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += lossfn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= numbatches #we're returning average loss on whatever batched data loader we've been given
    
    correct /= len(test_loader.dataset)
    model.train()
    return correct, test_loss

