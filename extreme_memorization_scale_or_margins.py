"""Allows experiments on network scale and margin.  Generates datasets
for figures 5, 10.

"""
import itertools
import sys
import math
from tqdm import tqdm

import pickle

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F 

sys.path.append("..")

from torch.optim import SGD

from util.nero import Nero
from util.data import get_data, get_data_k_class 
from util.trainer import generalized_multiclass_train_fullbatch
from util.scale_init import NetScaleInit
from matplotlib import cm
from generalization_bounds import bartlett_spectral_complexity

# Experiment toggle: -------------------------------
# which_experiment = 'controlling init scale'
# which_experiment = 'controlling margin'
which_experiment = 'controlling normalized margin'
# --------------------------------------------------

###### Start Code Blocks for different experiment parameter settings:
## init scale Block:
if which_experiment == 'controlling init scale':
    experiment_flag = 'init_scale'
    opt_string = 'SGD'
    constraints_string = 'False'
    epochs = 250000
    lr_decay = 0.999998
    lr = 0.0001
## End init scale Block

#margin Block:
if which_experiment == 'controlling margin':
    experiment_flag = 'margin_scale'
    opt_string = 'SGD'
    constraints_string = 'False'
    epochs = 50000
    lr_decay = 0.9998
    lr = 0.01
# End Margin Block


## Frob-normalized margin Block
if which_experiment == 'controlling normalized margin':
    experiment_flag = 'margin_scale'
    opt_string = 'Nero'
    constraints_string = 'True'
    epochs = 50000
    lr_decay = 0.9998
    lr = 0.01
# End Frob-normalized margin Block
###### End Code Blocks for different experiment parameter settings

fname = f"generalization_scale_{experiment_flag}_{opt_string}_{constraints_string}_constraints.pkl"


if constraints_string == 'False':
    constraints_flag = False
elif constraints_string == 'True':
    constraints_flag =  True


sigma_base = 0.0001
if experiment_flag == 'margin_scale':
    scale_vector = np.logspace(-4, 1, num=25)
    param_vector = scale_vector
    sigma_vector = None
    sigma = sigma_base
    true_data_label_scale = None
    rand_label_scale = None

elif experiment_flag == 'init_scale':   
    sigma = sigma_base
    sigma_vector = [sigma_base * 2**i for i in range(15)]

    true_data_label_scale = 1
    rand_label_scale = 1    
    param_vector = sigma_vector
    scale_vector = [1]

num_networks = len(param_vector)

# epochs = 100000
# lr  = 0.0001
depth = 2
width = 2048
k_classes = 10
num_train_examples = 1000 
# lr_decay = 0.99998

tqdm_flag = False

lr_vector = []
if opt_string == 'Nero':
    cur_opt = Nero
    optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
    optimizer_kwargs_to_save = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
elif opt_string == 'SGD':
    cur_opt = SGD
    optimizer_kwargs = {'lr':lr}
    optimizer_kwargs_to_save = {'lr':lr}
    if experiment_flag == 'init_scale':
        lr_vector = [lr  for i in range(len(sigma_vector))]

to_spect_norm = False

criterion = nn.MSELoss()
early_stop = False

params_set = {'num_networks': num_networks,
              'epochs': epochs,
              'lr': lr,
              'depth': depth,
              'width': width,
              'num_train_examples': num_train_examples,
              'lr_decay': lr_decay,
              'optimizer': cur_opt,
              'optimizer_kwargs': optimizer_kwargs_to_save,
              'true_data_label_scale': true_data_label_scale,
              'rand_label_scale': rand_label_scale,
              'sigma_vector': sigma_vector,
              'loss_function': criterion,
              'scale_vector': scale_vector,
              'experiment_flag': experiment_flag,
              'lr_vector': lr_vector
              }


true_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'X_norms': [],
                     'sigma': [],
                     'optimizer_kwargs': [],
                     'lr_decay': []
                     }

rand_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'X_norms': [],
                     'sigma': [],
                     'optimizer_kwargs': [],
                     'lr_decay': []
                     }

to_print = []
for net, cur_param in enumerate(tqdm(param_vector)):
    if experiment_flag == 'init_scale':
        sigma = cur_param
        cur_scale = 1
    elif experiment_flag == 'margin_scale':
        cur_scale = cur_param

    true_data_label_scale = cur_scale
    rand_data_label_scale = cur_scale

    full_batch_train_loader, train_loader, test_loader = get_data(
        num_train_examples=num_train_examples,
        batch_size=num_train_examples,
        random_labels=False,
        binary_digits=False)

    if opt_string == 'SGD' and experiment_flag == 'init_scale':
        lr_decay_dict = {
            0.1024: {'lr': 0.0001,
                         'lr_decay': 0.99998,
                         'epochs': 250000},
            0.2048: {'lr': 0.0001,
                         'lr_decay': 0.99998,
                         'epochs': 250000},
            0.4096: {'lr': 0.0001,
                         'lr_decay': 0.99998,
                         'epochs': 250000},
            0.8192: {'lr': 1e-5,
                         'lr_decay': 0.999998,
                         'epochs': 250000},
            1.6384: {'lr': 1e-5,
                         'lr_decay': 0.999998,
                         'epochs':250000}
        }
        if sigma in lr_decay_dict.keys():
            optimizer_kwargs['lr'] = lr_decay_dict[sigma]['lr']
            lr_decay = lr_decay_dict[sigma]['lr_decay']
            epochs = lr_decay_dict[sigma]['epochs']


    model = NetScaleInit(depth, width, k_classes, sigma)
    train_acc, test_acc, model, init_weights, outputs, correct_class_outputs,\
    other_class_outputs, X_norm, targets = \
        generalized_multiclass_train_fullbatch(model,
                                        criterion,
                                        train_loader,
                                        test_loader, 
                                        k_classes, 
                                        cur_opt, 
                                        optimizer_kwargs,
                                        epochs,
                                        lr_decay,
                                        to_spect_norm=False,
                                        label_scale=true_data_label_scale,
                                        return_init=True,
                                        return_margins=True,
                                        early_stop=early_stop,
                                        tqdm_flag = tqdm_flag)
    to_print.append(f'scale_init: {sigma}, lr: {optimizer_kwargs["lr"]}, train acc: {train_acc[-1]}')

    rand_full_batch_train_loader, rand_train_loader, rand_test_loader = get_data_k_class(
        num_train_examples=num_train_examples,
        batch_size=num_train_examples,
        random_labels=True,
        binary_digits=False,
        k_classes = k_classes)


    rand_model = NetScaleInit(depth, width, k_classes, sigma)
    rand_train_acc, rand_test_acc, rand_model, rand_init_weights, rand_outputs, rand_correct_class_outputs,\
    rand_other_class_outputs, rand_X_norm, rand_targets = \
        generalized_multiclass_train_fullbatch(rand_model,
                                        criterion,
                                        rand_train_loader,
                                        rand_test_loader, 
                                        k_classes, 
                                        cur_opt, 
                                        optimizer_kwargs,
                                        epochs,
                                        lr_decay,
                                        to_spect_norm=False,
                                        label_scale=rand_data_label_scale,
                                        return_init=True,
                                        return_margins=True,
                                        early_stop=early_stop,
                                        tqdm_flag=tqdm_flag)
    fro_norm = []
    fro_norm_rand = []
    spectral_norm = []
    spectral_norm_rand = []
    model.cpu()
    rand_model.cpu()
    for idx, params in enumerate(zip(model.parameters(), rand_model.parameters())):

        p, p_rand = params
        p = p.detach()
        p_rand = p_rand.detach()
        fro_norm.append(np.linalg.norm(p, ord='fro'))
        fro_norm_rand.append(np.linalg.norm(p_rand, ord='fro'))

        spectral_norm.append(np.linalg.norm(p, ord=2))
        spectral_norm_rand.append(np.linalg.norm(p_rand, ord=2))

    spect_complex = bartlett_spectral_complexity(model, ref_M=init_weights)
    rand_spect_complex = bartlett_spectral_complexity(rand_model, ref_M=rand_init_weights)

    optimizer_kwargs.pop('params', None)


    true_data_results['train_acc_list'].append(train_acc)
    true_data_results['test_acc_list'].append(test_acc)
    true_data_results['fro_norms'].append(fro_norm)
    true_data_results['spect_norms'].append(spectral_norm)
    true_data_results['correct_class_outputs'].append(correct_class_outputs)
    true_data_results['other_class_outputs'].append(other_class_outputs)
    true_data_results['bartlett_spect_complexity'].append(spect_complex)
    true_data_results['X_norms'].append(X_norm)
    true_data_results['sigma'].append(sigma)
    true_data_results['optimizer_kwargs'].append(optimizer_kwargs)
    true_data_results['lr_decay'].append(lr_decay)


    rand_data_results['train_acc_list'].append(rand_train_acc)
    rand_data_results['test_acc_list'].append(rand_test_acc)
    rand_data_results['fro_norms'].append(fro_norm_rand)
    rand_data_results['spect_norms'].append(spectral_norm_rand)
    rand_data_results['correct_class_outputs'].append(rand_correct_class_outputs)
    rand_data_results['other_class_outputs'].append(rand_other_class_outputs)
    rand_data_results['bartlett_spect_complexity'].append(rand_spect_complex)
    rand_data_results['X_norms'].append(rand_X_norm)
    rand_data_results['sigma'].append(sigma)
    rand_data_results['optimizer_kwargs'].append(optimizer_kwargs)
    rand_data_results['lr_decay'].append(lr_decay)




final_vals_dict = {'true_data_results': true_data_results,
                   'rand_data_results': rand_data_results}

final_vals_dict['parameters'] = params_set

for print_str in to_print:
    print(print_str)


f = open(fname, "wb")
pickle.dump(final_vals_dict, f)
f.close()
