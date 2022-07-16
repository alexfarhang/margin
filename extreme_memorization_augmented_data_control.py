"""Runs experiments from Figures 3, 7.  Can augment dataset with a form of 
attack set.

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
from torch.autograd.functional import vhp, hessian
from torchvision import datasets, transforms

sys.path.append("..")

from torch.optim import SGD
from util.nero import Nero
from util.data import get_data_attack_set
from util.trainer import SimpleNetMultiClass, generalized_multiclass_train_fullbatch

from matplotlib import cm
from generalization_bounds import bartlett_spectral_complexity


### Parameters to generate dataset for Figures 3, 7:
experiment_flag = 'margin_scale'
opt_string = 'Nero'
constraints_string = 'True'
###



if constraints_string == 'False':
    constraints_flag = False
elif constraints_string == 'True':
    constraints_flag =  True


if experiment_flag == 'margin_scale':
    param_size = 6
    scale_vector = np.logspace(-1, 1, num=param_size)
    param_vector = scale_vector
    sigma_vector = None
    sigma_base = 0.001
    sigma = sigma_base
    true_data_label_scale = None
    rand_label_scale = None

num_networks = len(param_vector)

epochs = 50000 
control_epochs = 50000
lr  = 0.01
depth = 2
width = 2048
k_classes = 10
num_train_examples = 500 
num_test_examples = 10000
num_attack_examples = 1500
lr_decay = 0.9999
tqdm_flag = False

if opt_string == 'Nero':
    cur_opt = Nero
    optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
    optimizer_kwargs_to_save = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
elif opt_string == 'SGD':
    cur_opt = SGD
    optimizer_kwargs = {'lr':lr}
    optimizer_kwargs_to_save = {'lr':lr}


to_spect_norm = False

criterion = nn.MSELoss()
early_stop = False

params_set = {'num_networks': num_networks,
              'epochs': epochs,
              'lr': lr,
              'depth': depth,
              'width': width,
              'num_train_examples': num_train_examples,
              'num_test_examples': num_test_examples,
              'num_attack_examples': num_attack_examples,
              'lr_decay': lr_decay,
              'optimizer': cur_opt,
              'optimizer_kwargs': optimizer_kwargs_to_save,
              'true_data_label_scale': true_data_label_scale,
              'rand_label_scale': rand_label_scale,
              'sigma_vector': sigma_vector,
              'loss_function': criterion,
              'scale_vector': scale_vector,
              'experiment_flag': experiment_flag,
              'k_classes': k_classes,
              'control_epochs': control_epochs
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
                     }

control_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'X_norms': [],
                     'sigma': []
                     }

for net, cur_param in enumerate(tqdm(param_vector)):
    if experiment_flag == 'init_scale':
        sigma = cur_param
        cur_scale = 1
    elif experiment_flag == 'margin_scale':
        cur_scale = cur_param

    true_data_label_scale = cur_scale
    control_data_label_scale = cur_scale

    full_batch_train_loader_attack, full_batch_test_loader, train_loader, test_loader, full_batch_control_train_loader, control_train_loader = get_data_attack_set(
        num_train_examples=num_train_examples, 
        num_attack_examples=num_attack_examples, 
        num_test_examples=num_test_examples, 
        batch_size=num_train_examples, 
        k_classes=k_classes, 
        control=True)
    
    full_batch_train_loader = full_batch_train_loader_attack




    model = SimpleNetMultiClass(depth, width, k_classes)
    train_acc, test_acc, model, init_weights, outputs, correct_class_outputs,\
    other_class_outputs, X_norm, targets = \
        generalized_multiclass_train_fullbatch(model,
                                        criterion,
                                        full_batch_train_loader,
                                        full_batch_test_loader, 
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


    # Now the NN trained without the distorted data
    control_model = SimpleNetMultiClass(depth, width, k_classes)

    control_train_acc, control_test_acc, control_model, control_init_weights, control_outputs, control_correct_class_outputs,\
    control_other_class_outputs, control_X_norm, control_targets = \
        generalized_multiclass_train_fullbatch(control_model,
                                        criterion,
                                        full_batch_control_train_loader,
                                        full_batch_test_loader, 
                                        k_classes, 
                                        cur_opt, 
                                        optimizer_kwargs,
                                        control_epochs,
                                        lr_decay,
                                        to_spect_norm=False,
                                        label_scale=control_data_label_scale,
                                        return_init=True,
                                        return_margins=True,
                                        early_stop=early_stop,
                                        tqdm_flag = tqdm_flag)


    fro_norm = []
    fro_norm_control = []
    spectral_norm = []
    spectral_norm_control = []
    model.cpu()
    control_model.cpu()
    for idx, params in enumerate(zip(model.parameters(), control_model.parameters())):

        p, p_control = params
        p = p.detach()
        p_control = p_control.detach()
        fro_norm.append(np.linalg.norm(p, ord='fro'))
        fro_norm_control.append(np.linalg.norm(p_control, ord='fro'))
        spectral_norm.append(np.linalg.norm(p, ord=2))
        spectral_norm_control.append(np.linalg.norm(p_control, ord=2))

    spect_complex = bartlett_spectral_complexity(model, ref_M=init_weights)
    control_spect_complex = bartlett_spectral_complexity(control_model, ref_M = control_init_weights)


    true_data_results['train_acc_list'].append(train_acc)
    true_data_results['test_acc_list'].append(test_acc)
    true_data_results['fro_norms'].append(fro_norm)
    true_data_results['spect_norms'].append(spectral_norm)
    true_data_results['correct_class_outputs'].append(correct_class_outputs)
    true_data_results['other_class_outputs'].append(other_class_outputs)
    true_data_results['bartlett_spect_complexity'].append(spect_complex)
    true_data_results['X_norms'].append(X_norm)
    true_data_results['sigma'].append(sigma)

    control_data_results['train_acc_list'].append(control_train_acc)
    control_data_results['test_acc_list'].append(control_test_acc)
    control_data_results['fro_norms'].append(fro_norm_control)
    control_data_results['spect_norms'].append(spectral_norm_control)
    control_data_results['correct_class_outputs'].append(control_correct_class_outputs)
    control_data_results['other_class_outputs'].append(control_other_class_outputs)
    control_data_results['bartlett_spect_complexity'].append(control_spect_complex)
    control_data_results['X_norms'].append(control_X_norm)
    control_data_results['sigma'].append(sigma)





final_vals_dict = {'true_data_results': true_data_results,
                   'control_data_results': control_data_results
                   }

final_vals_dict['parameters'] = params_set


f = open(f"attack_set_with_control_dif_epochs_{opt_string}_{constraints_string}_constraints_{epochs}_epochs_{lr_decay}_lrdecay_{depth}_depth.pkl", "wb")
pickle.dump(final_vals_dict, f)
f.close()
