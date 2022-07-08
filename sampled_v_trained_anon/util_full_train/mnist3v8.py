import math
import numpy as np
import torch
from torchvision import datasets, transforms

trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

testset = datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

def get_data(num_train_examples, num_test_examples, batch_size, random_labels, binary_digits, mnist_first_class, mnist_second_class, full_dataset=False):
    if binary_digits:
        train_superset = []
        for data in trainset:
            if data[1] == mnist_first_class or data[1] == mnist_second_class:
                train_superset.append( data )
        test_superset = []
        for data in testset:
            if data[1] == mnist_first_class or data[1] == mnist_second_class:
                test_superset.append( data )
    else:
        train_superset = trainset
        test_superset = testset

    if full_dataset:
        indices = np.random.permutation(len(train_superset))
    else:    
        indices = np.random.permutation(len(train_superset))[0:num_train_examples]

    train_subset = torch.utils.data.Subset(train_superset, indices)

    if num_test_examples is None:
        num_test_examples = len(test_superset)

    if full_dataset:
        indices = np.random.permutation(len(test_superset))
    else:
        indices = np.random.permutation(len(test_superset))[0:num_test_examples]
    
    test_subset = torch.utils.data.Subset(test_superset, indices)

    if random_labels:
        random_train_subset = []
        for data in train_subset:
            random_train_subset.append( ( data[0], torch.randint(low=0,high=2,size=(1,)).item() ) )
        train_subset = random_train_subset

        random_test_subset = []
        for data in test_subset:
            random_test_subset.append( ( data[0], torch.randint(low=0,high=2,size=(1,)).item() ) )
        test_subset = random_test_subset

    full_batch_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    full_batch_test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    # train_loader = torch.utils.data.DataLoader(train_subset,  batch_size=batch_size, shuffle=True)
    # test_loader  = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return full_batch_train_loader, full_batch_test_loader

def normalize_data_even_odd(data, target):
    # [num_training_points, 1, width, height] => [num_training_points, width*height]
    # [5, 1, 28, 28] => [5, 784]
    data = data.view(data.shape[0],-1)

    # computes the per-training-point frobenius norm  (so the frob norm of each training point matrix will be 1.0)
    data /= data.norm(dim=1).unsqueeze(dim=1)

    # multiplies everything by sqrt(width*height) (so now the frob norm of each training point will be sqrt(width*height))
    data *= math.sqrt(data.shape[1])

    # turns into a binary classification problem i.e [3, 6, 6, 6, 0] -> [ 1, -1, -1, -1, -1]
    target = target%2*2-1
    return data, target