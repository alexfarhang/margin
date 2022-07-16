"""Code to generate data from Figures 1 and 2.  Adds norm constraints and allows
unequal label scale.
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
from util.data import get_data, normalize_data, get_data_k_class
from util.trainer import SimpleNetMultiClass, generalized_multiclass_train_fullbatch

from generalization_bounds import bartlett_spectral_complexity

# ### To Generate Figure 1:
# experiment_flag = 'no_constraint'
# label_scale_string ='unequal'
# ### End Figure 1

### To Generate Figure 2:
experiment_flag = 'fr_constraint'
label_scale_string = 'unequal'
### End Figure 2


# label scales
if label_scale_string == "equal":
    true_data_label_scale = 1
    rand_label_scale = 1
elif label_scale_string == 'unequal':
    true_data_label_scale = 1
    rand_label_scale = 100
    if experiment_flag == 'no_constraint':
        rand_label_scale = 10


num_networks = 3
epochs = 1000
lr  = 0.01
depth = 5
width = 5000
num_train_examples = 1000
k_classes = 10
lr_decay = 0.999
early_stop = False
tqdm_flag = False
delta = 0.001

to_spect_norm = False

if experiment_flag == 'spectral_constraint':
    opt_string = 'SGD'
    to_spect_norm = True
elif experiment_flag == 'fr_constraint':
    opt_string = 'Nero'
    constraints_flag = True
elif experiment_flag == 'no_constraint':
    opt_string = 'SGD'


if opt_string == 'Nero':
    cur_opt = Nero
    optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
    optimizer_kwargs_to_save = {'lr':lr, 'beta':0.999, 'constraints':constraints_flag}
elif opt_string == 'SGD':
    cur_opt = SGD
    optimizer_kwargs = {'lr': lr}
    optimizer_kwargs_to_save = {'lr': lr}



gamma_array = [10**gamma_mod for gamma_mod in range(0,11,2)]
params_set = {'num_networks': num_networks,
              'epochs': epochs,
              'lr': lr,
              'depth': depth,
              'width': width,
              'k_classes': k_classes,
              'num_train_examples': num_train_examples,
              'lr_decay': lr_decay,
              'bartlett_gamma_array': gamma_array,
              'delta': delta,
              'optimizer': cur_opt,
              'optimizer_kwargs_to_save': optimizer_kwargs_to_save,
              'true_data_label_scale': true_data_label_scale,
              'rand_label_scale': rand_label_scale
              }

criterion = nn.MSELoss()

true_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'bartlett_bound': [],
                     'min_bound': [],
                     'X_norms': [],
                     'bartlett_spect_complexity_M0': [],
                     'bound_dicts': [],
                     'gamma_array': []
                     }

rand_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'bartlett_bound': [],
                     'min_bound': [],
                     'X_norms': [],
                     'bartlett_spect_complexity_M0': [],
                     'bound_dicts': [],
                     'gamma_array': []
                     }

for net in tqdm(range(num_networks)):
    cur_gamma_array = [gamma for gamma in gamma_array]
    # Load and Train True data model (10 class)
    full_batch_train_loader, train_loader, test_loader = get_data(
        num_train_examples=num_train_examples,
        batch_size=num_train_examples,
        random_labels=False,
        binary_digits=False)

    model = SimpleNetMultiClass(depth, width, k_classes)
    train_acc, test_acc, model, init_weights, outputs, correct_class_outputs, other_class_outputs, X_norm, targets = \
        generalized_multiclass_train_fullbatch(model,
                                    criterion,
                                    train_loader,
                                    test_loader, 
                                    k_classes, 
                                    cur_opt, 
                                    optimizer_kwargs,
                                    epochs,
                                    lr_decay,
                                    to_spect_norm=to_spect_norm,
                                    label_scale=true_data_label_scale,
                                    return_init=True,
                                    return_margins=True,
                                    early_stop=early_stop)
    true_margins = correct_class_outputs - other_class_outputs
    true_margins = torch.Tensor([x for x in true_margins if x > 0])
    true_min_margin = min(true_margins)


    
    ########
    # Load and Train random model

    rand_full_batch_train_loader, rand_train_loader, rand_test_loader = get_data_k_class(
        num_train_examples=num_train_examples,
        batch_size=num_train_examples,
        random_labels=True,
        binary_digits=False,
        k_classes = k_classes)

    rand_model = SimpleNetMultiClass(depth, width, k_classes)
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
                                        to_spect_norm=to_spect_norm,
                                        label_scale=rand_label_scale,
                                        return_init=True,
                                        return_margins=True,
                                        early_stop=early_stop,
                                        tqdm_flag=tqdm_flag)
                                        
    print(f"true train acc: {train_acc[-1]}")
    print(f"rand train acc: {rand_train_acc[-1]}")
    rand_margins = rand_correct_class_outputs - rand_other_class_outputs
    rand_margins = torch.Tensor([x for x in rand_margins if x > 0])
    if len(rand_margins) == 0:
        rand_min_margin = true_min_margin
    else:
        rand_min_margin = min(rand_margins)


    fro_norm = []
    fro_norm_rand = []
    spectral_norm = []
    spectral_norm_rand = []
    model.cpu()
    rand_model.cpu()
    cur_gamma_array.append(true_min_margin.item())
    cur_gamma_array.append(rand_min_margin.item())

    for idx, params in enumerate(zip(model.parameters(), rand_model.parameters())):

        p, p_rand = params
        p = p.detach()
        p_rand = p_rand.detach()
        fro_norm.append(np.linalg.norm(p, ord='fro'))
        fro_norm_rand.append(np.linalg.norm(p_rand, ord='fro'))

        spectral_norm.append(np.linalg.norm(p, ord=2))
        spectral_norm_rand.append(np.linalg.norm(p_rand, ord=2))

    spect_complex = bartlett_spectral_complexity(model, ref_M=init_weights)
    spect_complex_M0 = bartlett_spectral_complexity(model, ref_M = None)

    rand_spect_complex = bartlett_spectral_complexity(rand_model, ref_M=rand_init_weights)
    rand_spect_complex_M0 = bartlett_spectral_complexity(rand_model, ref_M = None)
    
    cur_bound_dict = {}
    rand_cur_bound_dict = {}


    true_data_results['train_acc_list'].append(train_acc)
    true_data_results['test_acc_list'].append(test_acc)
    true_data_results['fro_norms'].append(fro_norm)
    true_data_results['spect_norms'].append(spectral_norm)
    true_data_results['correct_class_outputs'].append(correct_class_outputs)
    true_data_results['other_class_outputs'].append(other_class_outputs)
    true_data_results['bartlett_spect_complexity'].append(spect_complex)
    true_data_results['bartlett_spect_complexity_M0'].append(spect_complex_M0)
    true_data_results['X_norms'].append(X_norm)

    rand_data_results['train_acc_list'].append(rand_train_acc)
    rand_data_results['test_acc_list'].append(rand_test_acc)
    rand_data_results['fro_norms'].append(fro_norm_rand)
    rand_data_results['spect_norms'].append(spectral_norm_rand)
    rand_data_results['correct_class_outputs'].append(rand_correct_class_outputs)
    rand_data_results['other_class_outputs'].append(rand_other_class_outputs)
    rand_data_results['bartlett_spect_complexity'].append(rand_spect_complex)
    rand_data_results['X_norms'].append(rand_X_norm)



final_vals_dict = {'true_data_results': true_data_results,
                   'rand_data_results': rand_data_results}

final_vals_dict['parameters'] = params_set



f = open(f"spect_complexity_experiments_{experiment_flag}_{label_scale_string}_labels_{rand_label_scale}_randlabelscale_{epochs}_epochs.pkl", "wb")
pickle.dump(final_vals_dict, f)
f.close()
