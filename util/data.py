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

def get_data(num_train_examples, batch_size, random_labels, binary_digits):

    if binary_digits:
        train_superset = []
        for data in trainset:
            if data[1] < 2:
                train_superset.append( data )
        test_superset = []
        for data in testset:
            if data[1] < 2:
                test_superset.append( data )
    else:
        train_superset = trainset
        test_superset = testset

    indices = np.random.permutation(len(train_superset))[0:num_train_examples]
    train_subset = torch.utils.data.Subset(train_superset, indices)

    if random_labels:
        random_train_subset = []
        for data in train_subset:
            random_train_subset.append( ( data[0], torch.randint(low=0,high=2,size=(1,)).item() ) )
        train_subset = random_train_subset

    full_batch_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_subset,  batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_superset, batch_size=batch_size, shuffle=False)

    return full_batch_train_loader, train_loader, test_loader


def get_data_k_class(num_train_examples, batch_size, random_labels, binary_digits, k_classes):
# """Needed for random k class labels, as the other will only give binary labels"""
    if binary_digits:
        train_superset = []
        for data in trainset:
            if data[1] < 2:
                train_superset.append( data )
        test_superset = []
        for data in testset:
            if data[1] < 2:
                test_superset.append( data )
    else:
        train_superset = trainset
        test_superset = testset

    indices = np.random.permutation(len(train_superset))[0:num_train_examples]
    train_subset = torch.utils.data.Subset(train_superset, indices)

    if random_labels:
        random_train_subset = []
        for data in train_subset:
            random_train_subset.append( ( data[0], torch.randint(low=0,high=k_classes,size=(1,)).item() ) )
        train_subset = random_train_subset

    full_batch_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_subset,  batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_superset, batch_size=batch_size, shuffle=False)

    return full_batch_train_loader, train_loader, test_loader

def normalize_data(data, target):
    data = data.view(data.shape[0],-1)
    data /= data.norm(dim=1).unsqueeze(dim=1)
    data *= math.sqrt(data.shape[1])
    target = target%2*2-1
    return data, target


def normalize_data_10_class(data, target):
    data = data.view(data.shape[0],-1)
    data /= data.norm(dim=1).unsqueeze(dim=1)
    data *= math.sqrt(data.shape[1])
    # target = target%2*2-1
    return data, target


def get_data_attack_set(num_train_examples, num_attack_examples, num_test_examples, batch_size, k_classes, control=False):
# """For extreme memorization augmented train set experiment.  Appends randomlyu labeled train set (not included in the original) as attack set.  The test set remains correctly labeled."""
# If control=True, also returns the unaugmented train set 
    
    train_superset = trainset
    test_superset = testset

    perm = np.random.permutation(len(train_superset))
    indices_attack = perm[num_train_examples:num_train_examples + num_attack_examples]
    indices_train = perm[0:num_train_examples]
    train_subset = torch.utils.data.Subset(train_superset, indices_train)
    attack_train_subset = torch.utils.data.Subset(train_superset, indices_attack)
        

    if num_test_examples is None:
        num_test_examples = len(test_superset)

    indices_test = np.random.permutation(len(test_superset))[0:num_test_examples]
    test_subset = torch.utils.data.Subset(test_superset, indices_test)

 
    random_train_subset = []
    for data in attack_train_subset:
        random_train_subset.append( ( data[0], torch.randint(low=0,high=k_classes,size=(1,)).item() ) )
    train_subset = list(train_subset)
    train_subset = train_subset + random_train_subset
        # Now train_subset is a combination of the true train points and the random labeled test points
        # the test_subset is still correctly labeled
    

    full_batch_train_loader_attack = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    full_batch_test_loader = torch.utils.data.DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_subset,  batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    if control:
        train_control_subset = torch.utils.data.Subset(train_superset, indices_train)
        full_batch_control_train_loader = torch.utils.data.DataLoader(train_control_subset, batch_size = len(train_control_subset), shuffle=False)
        control_train_loader = torch.utils.data.DataLoader(train_control_subset, batch_size=batch_size, shuffle=True)
        return full_batch_train_loader_attack, full_batch_test_loader, train_loader, test_loader, full_batch_control_train_loader, control_train_loader
    else:
        return full_batch_train_loader, full_batch_test_loader, train_loader, test_loader


def get_data_augmented_k_classes(num_train_examples, num_test_examples, batch_size, k_classes, control=False):
# """For extreme memorization augmented train set experiment.  Adds test points to the train set, but randomizes their labels.  The test set remains correctly labeled."""
# If control=True, also returns the unaugmented train set 
    
    train_superset = trainset
    test_superset = testset

    # perm = np.random.permutation(len(train_superset))
    # indices_attack = perm[num_train_examples:num_train_examples + num_attack_examples]
    indices_train = perm[0:num_train_examples]
    train_subset = torch.utils.data.Subset(train_superset, indices_train)
        

    if num_test_examples is None:
        num_test_examples = len(test_superset)

    indices = np.random.permutation(len(test_superset))[0:num_test_examples]
    test_subset = torch.utils.data.Subset(test_superset, indices)

 
    random_test_subset = []
    for data in test_subset:
        random_test_subset.append( ( data[0], torch.randint(low=0,high=k_classes,size=(1,)).item() ) )
    train_subset = list(train_subset)
    train_subset = train_subset + random_test_subset
        # Now train_subset is a combination of the true train points and the random labeled test points
        # the test_subset is still correctly labeled
    

    full_batch_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
    full_batch_test_loader = torch.utils.data.DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_subset,  batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    if control:
        train_control_subset = torch.utils.data.Subset(train_superset, indices_train)
        full_batch_control_train_loader = torch.utils.data.DataLoader(train_control_subset, batch_size = len(train_control_subset), shuffle=False)
        control_train_loader = torch.utils.data.DataLoader(train_control_subset, batch_size=batch_size, shuffle=True)
        return full_batch_train_loader, full_batch_test_loader, train_loader, test_loader, full_batch_control_train_loader, control_train_loader
    else:
        return full_batch_train_loader, full_batch_test_loader, train_loader, test_loader
