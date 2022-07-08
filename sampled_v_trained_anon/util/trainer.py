import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nero import Nero
from util.data import normalize_data_even_odd

# from tqdm import tqdm
tqdm = lambda x: x

class AdaptiveConvNet(nn.Module):
    def __init__(self, conv_depth, conv_width, lin_depth, lin_width):
        super(AdaptiveConvNet, self).__init__()
        self.init_conv = nn.Conv2d(3, conv_width, 5, bias=False)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv_layers = nn.ModuleList([nn.Conv2d(conv_width, conv_width, 5, bias=False) for _ in range(conv_depth-2)])

        self.conv_to_lin = nn.Linear(conv_width * 24 * 24, lin_width, bias=False)
        self.lin_layers = nn.ModuleList([nn.Linear(lin_width, lin_width, bias=False) for _ in range(lin_depth-2)])
        self.lin_to_output = nn.Linear(lin_width, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.init_conv(x))
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.conv_to_lin(x))

        for layer in self.lin_layers:
            x = F.relu(layer(x))

        x = self.lin_to_output(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 1, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, initial, depth, width):
        super(SimpleNet, self).__init__()

        self.initial = nn.Linear(initial, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)


def train_network(train_loader, test_loader, depth, width, init_lr, decay, break_on_fit=True, target_scale=1.0):
    
    model = SimpleNet(depth, width).cuda()
    optim = Nero(model.parameters(), lr=init_lr)
    model_init = copy.deepcopy(model)

    lr_lambda = lambda x: decay**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    train_acc_list = []
    train_acc = 0

    for epoch in tqdm(range(50)):
        model.train()

        for data, target in train_loader:
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data_even_odd(data, target)

            y_pred = model(data).squeeze()
            loss = (y_pred - target_scale*target).norm()

            model.zero_grad()
            loss.backward()
            optim.step()

        lr_scheduler.step()

        model.eval()
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data_even_odd(data, target)

            y_pred = model(data).squeeze()
            correct += (target.float() == y_pred.sign()).sum().item()
            total += target.shape[0]

        train_acc = correct/total
        train_acc_list.append(train_acc)

        if break_on_fit and train_acc == 1.0: break

    model.eval()
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = (data.cuda(), target.cuda())
        data, target = normalize_data(data, target)
        
        y_pred = model(data).squeeze()
        correct += (target.float() == y_pred.sign()).sum().item()
        total += target.shape[0]

    test_acc = correct/total
  
    return train_acc_list, test_acc, model, model_init
