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
# from torch.autograd.functional import vhp, hessian
# from torchvision import datasets, transforms

sys.path.append("..")

from torch.optim import SGD
# from torch.optim.optimizer import Optimizer
# from fromage import Fromage
from util.nero import Nero
from util.data import get_data, get_data_k_class #,normalize_data, normalize_data_10_class
# from util.trainer import SimpleNet, SimpleNetMultiClass, train_network, train_network_multiclass, train_network_multiclass_combined, train_network_multiclass_scale_label
from util.trainer import generalized_multiclass_train_fullbatch#, train_network_multiclass_scale_label_input_net, generalized_multiclass_train, generalized_multiclass_train_fullbatch
# from util.hessians import *
from util.scale_init import NetScaleInit
from matplotlib import cm
from generalization_bounds import *

# Experiment toggle: -------------------------------
which_experiment = 'controlling init scale'
# which_experiment = 'controlling margin'
# which_experiment = 'controlling normalized margin'
# --------------------------------------------------

###### Start Code Blocks for different experiment parameter settings:
## init scale Block:
if which_experiment == 'controlling init scale':
    experiment_flag = 'init_scale'
    opt_string = 'SGD'
    constraints_string = 'False'
    epochs = 250000
    lr_decay = 0.999998
## End init scale Block

#margin Block:
if which_experiment == 'controlling margin':
    experiment_flag = 'margin_scale'
    opt_string = 'SGD'
    constraints_flag = 'False'
    epochs = 50000
    lr_decay = 0.9998
# End Margin Block


## Frob-normalized margin Block
if which_experiment == 'controlling normalized margin':
    experiment_flag = 'margin_scale'
    opt_string = 'Nero'
    constraints_flag = 'True'
    epochs = 50000
    lr_decay = 0.9998
# End Frob-normalized margin Block
###### End Code Blocks for different experiment parameter settings

fname = f"generalization_scale_{experiment_flag}_{opt_string}_{constraints_string}_constraints.pkl"

# specify the types of experiments here
# experiment_flag = 'init_scale'
# experiment_flag = 'margin_scale'

# opt_string = 'Nero'
# opt_string = 'SGD'
# constraints_string = 'False' # Really only for Nero
# constraints_string = 'True'

if constraints_string == 'False':
    constraints_flag = False
elif constraints_string == 'True':
    constraints_flag =  True


sigma_base = 0.0001
if experiment_flag == 'margin_scale':
    # scale_vector = np.logspace(-2, 1, num=15)
    scale_vector = np.logspace(-4, 1, num=25)

    param_vector = scale_vector
    sigma_vector = None
    # sigma_base = 0.001
    sigma = sigma_base
    true_data_label_scale = None
    rand_label_scale = None

elif experiment_flag == 'init_scale':   
    # sigma_base = 0.001
    # sigma_base = 0.0001
    # sigma_vector = sigma_base
    sigma = sigma_base
    # sigma_vector = [sigma_base * 2**i for i in range(11)]
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
# lr_decay = 1
# lr_decay = 0.99998

tqdm_flag = False

lr_vector = []
if opt_string == 'Nero':
    cur_opt = Nero
    optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
    optimizer_kwargs_to_save = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
elif opt_string == 'SGD':
    cur_opt = SGD
    optimizer_kwargs = {'lr':lr}#, 'beta':0.999, 'constraints':True}
    optimizer_kwargs_to_save = {'lr':lr}#, 'beta':0.999, 'constraints':True}
    if experiment_flag == 'init_scale':
        lr_vector = [lr  for i in range(len(sigma_vector))]#* 4**i for i in range(len(sigma_vector))]
# elif opt_string == 'ScaledNero':
#     cur_opt = ScaledNero
#     optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':True}
#     optimizer_kwargs_to_save = {'lr':lr, 'beta':0.999, 'constraints':True}# cur_opt = Nero

to_spect_norm = False
# to_train_on_rand = Falsw

criterion = nn.MSELoss()
early_stop = False


# delta = 0.05
# gamma_array = [10**gamma_mod for gamma_mod in range(0,11,2)]
params_set = {'num_networks': num_networks,
              'epochs': epochs,
              'lr': lr,
              'depth': depth,
              'width': width,
              'num_train_examples': num_train_examples,
              'lr_decay': lr_decay,
            #   'bartlett_gamma_array': gamma_array,
            #   'delta': delta,
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
# for net in tqdm(range(num_networks)):
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
        # optimizer_kwargs['lr'] = lr_vector[net]
        # print(lr_vector[net])

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
    # print(f'scale_init: {sigma}, lr: {optimizer_kwargs["lr"]}, train acc: {train_acc[-1]}')
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


# f = open("generalization_extreme_memorization_margin_scale.pkl", "wb")
# f = open(f"lr_specific_faster_longer_generalization_extreme_memorization_{experiment_flag}_{opt_string}_{constraints_string}_constraints_{epochs}_epochs_{lr_decay}_lrdecay_{depth}_depth.pkl", "wb")
f = open(fname, "wb")
pickle.dump(final_vals_dict, f)
f.close()
