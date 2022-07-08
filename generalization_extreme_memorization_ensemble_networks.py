# # BLOCK HERE
# """ Experiments examining the effect of empirical margin on network averaging
# """
# import itertools
# import sys
# import math
# from tqdm import tqdm

# import pickle

# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# from torchvision import datasets, transforms

# sys.path.append("..")

# from torch.optim import SGD
# from torch.optim.optimizer import Optimizer
# from fromage import Fromage
# from util.nero import Nero
# from util.data import get_data, normalize_data, normalize_data_10_class, get_data_k_class
# from util.trainer import SimpleNet, SimpleNetMultiClass, train_network, train_network_multiclass, train_network_multiclass_combined, train_network_multiclass_scale_label
# from util.trainer import train_network_multiclass_scale_label_input_net, generalized_multiclass_train
# from util.hessians import *
# from util.scale_init import NetScaleInit
# from matplotlib import cm
# from generalization_bounds import *
# #TODO: This code does not work with training networks on randomly shuffled data.
# # It will only correctly work with full batch trainng, else need to find new way
# # to sum outputs

# # scale_vector = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 10, 20]
# scale_vector = [0.01, 1]
# scale_vector.reverse()

# num_networks = 2#10
# epochs = 300
# lr  = 0.01
# depth = 5
# width = 2048
# k_classes = 10
# num_train_examples = 2000
# lr_decay = 0.99
# # to_binarize_data = True
# tqdm_flag = True
# tqdm_ = lambda x: x
# cur_opt = Nero
# # cur_opt = SGD
# optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':True}
# early_stop = True

# to_spect_norm = False
# # to_spect_norm = True

# # criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

# # This does nothing.  Never trained on random data
# to_train_rand_data = False


# # delta = 0.05
# # gamma_array = [10**gamma_mod for gamma_mod in range(0,11,2)]
# params_set = {'num_networks': num_networks,
#               'epochs': epochs,
#               'lr': lr,
#               'depth': depth,
#               'width': width,
#               'num_train_examples': num_train_examples,
#               'lr_decay': lr_decay,
#             #   'bartlett_gamma_array': gamma_array,
#             #   'delta': delta,
#               'optimizer': cur_opt,
#             #   'true_data_label_scale': true_data_label_scale,
#             #   'rand_label_scale': rand_label_scale,
#             #   'sigma_vector': sigma_vector,
#               'loss_function': criterion,
#               'scale_vector': scale_vector
#               }


# all_exp_dicts = []
# for exp_idx, cur_scale in tqdm(enumerate(scale_vector)):
#     # Train all networks in an experiment on the same data
#     full_batch_train_loader, train_loader, test_loader = get_data(
#         num_train_examples=num_train_examples,
#         batch_size=num_train_examples,
#         random_labels=False,
#         binary_digits=False)

#     # models_to_sum = []
#     model_outputs = {'train': [],
#                      'test': []
#                      }

#     models_dicts = []
#     exp_dict = {'models_dicts': models_dicts,
#             'label_scale': cur_scale,
#             'summed_train_acc':[],
#             'summed_test_acc': []
#             }
#     # train_outputs_to_sum = []
#     test_outputs_to_sum = []
#     for net in tqdm(range(num_networks)):
#         true_data_results = {'train_acc_list': [],
#                      'test_acc_list': [],
#                      'correct_class_outputs': [],
#                      'other_class_outputs': [],
#                      'fro_norms': [],
#                      'spect_norms': [],
#                      'bartlett_spect_complexity': [],
#                      'X_norms': []
#         }

#         true_data_label_scale = cur_scale
#         rand_data_label_scale = cur_scale

#         model = SimpleNetMultiClass(depth, width, k_classes)
#         train_acc, test_acc, model, init_weights, outputs, correct_class_outputs,\
#         other_class_outputs, X_norm, targets = \
#             generalized_multiclass_train(model,
#                                          criterion,
#                                          train_loader,
#                                          test_loader, 
#                                          k_classes, 
#                                          cur_opt, 
#                                          optimizer_kwargs,
#                                          epochs,
#                                          lr_decay,
#                                          to_spect_norm=False,
#                                          label_scale=true_data_label_scale,
#                                          return_init=True,
#                                          return_margins=True,
#                                          early_stop=early_stop)
        
#         fro_norm = []
#         spectral_norm = []
#         # model.cpu()
    
#         for idx, params in enumerate(model.parameters()):

#             p = params
#             # p = p.detach()
#             p = p.detach().cpu()
#             fro_norm.append(np.linalg.norm(p, ord='fro'))
#             spectral_norm.append(np.linalg.norm(p, ord=2))
#         spect_complex = bartlett_spectral_complexity(model, ref_M=init_weights)



#         true_data_results['train_acc_list'].append(train_acc)
#         true_data_results['test_acc_list'].append(test_acc)
#         true_data_results['fro_norms'].append(fro_norm)
#         true_data_results['spect_norms'].append(spectral_norm)
#         true_data_results['correct_class_outputs'].append(correct_class_outputs)
#         true_data_results['other_class_outputs'].append(other_class_outputs)
#         true_data_results['bartlett_spect_complexity'].append(spect_complex)
#         true_data_results['X_norms'].append(X_norm)
#         # true_data_results['sigma'].append(sigma)

#         models_dicts.append(true_data_results)

#         if net == 0:
#             train_outputs_to_sum = outputs
#         else:
#             train_outputs_to_sum += outputs

#         # Here compute test acc
#         with torch.no_grad():
#             # compute model output test acc to later sum
#             total = 0
#             correct = 0
#             for data_idx, (data, target) in enumerate(test_loader):
#                     data, target = (data.cuda(), target.cuda())
#                     if k_classes == 10:
#                         data, target = normalize_data_10_class(data, target)
#                     else:
#                         # normalize_data trains even/odd binary 
#                         data, target = normalize_data(data, target)
#                     # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#                         if k_classes == 2:
#                             target = (target + 1).true_divide(2)

#                     target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#                     target = target.cuda()
#                     y_pred_test_outputs = model(data).squeeze()
#                     if net == 0:
#                         test_outputs_to_sum.append(y_pred_test_outputs)
#                     else:
#                         test_outputs_to_sum[data_idx] += y_pred_test_outputs
#         del model

#     # test_outputs_to_sum here check accuracy
#     total = 0
#     correct = 0
#     for data_idx, (data, target) in enumerate(test_loader):
#         data, target = (data.cuda(), target.cuda())
#         if k_classes == 10:
#             data, target = normalize_data_10_class(data, target)
#         else:
#             data, target = normalize_data(data, target)
#             if k_classes == 2:
#                 target = (target + 1).true_divide(2)
#         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#         target = target.cuda()
#         y_pred_summed_test_outputs = test_outputs_to_sum[data_idx]
#         _, pred_indices = train_outputs_to_sum.cuda().max(dim=1)
#         _, target_indices = target.max(dim=1)
#         correct += (pred_indices == target_indices).sum().item()
#         total += target.shape[0]
#     summed_test_acc = correct/total

#     # train here
#     with torch.no_grad():
#         for data_idx, (data, target) in enumerate(train_loader):
#             total = 0
#             correct = 0
#             for data_idx, (data, target) in enumerate(train_loader):
#                     data, target = (data.cuda(), target.cuda())
#                     if k_classes == 10:
#                         data, target = normalize_data_10_class(data, target)
#                     else:
#                         # normalize_data trains even/odd binary 
#                         data, target = normalize_data(data, target)
#                     # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#                         if k_classes == 2:
#                             target = (target + 1).true_divide(2)
#                     target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#                     target = target.cuda()
#                     _, pred_indices = train_outputs_to_sum.cuda().max(dim=1)
#                     _, target_indices = target.max(dim=1)
#                     correct += (pred_indices == target_indices).sum().item()
#                     total += target.shape[0]
#         summed_train_acc = correct / total

       
#     # with torch.no_grad():
#     #     # compute summed model output test acc
#     #     total = 0
#     #     correct = 0
#     #     for data_idx, (data, target) in enumerate(test_loader):
#     #             data, target = (data.cuda(), target.cuda())
#     #             if k_classes == 10:
#     #                 data, target = normalize_data_10_class(data, target)
#     #             else:
#     #                 # normalize_data trains even/odd binary 
#     #                 data, target = normalize_data(data, target)
#     #             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#     #                 if k_classes == 2:
#     #                     target = (target + 1).true_divide(2)

#     #             target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#     #             target = target.cuda()
#     #             for idx, cur_model in enumerate(models_to_sum):
#     #                 cur_model.cuda()
#     #                 y_pred = cur_model(data).squeeze()
#     #                 if idx == 0:
#     #                     test_outputs = y_pred
#     #                 else:
#     #                     test_outputs += y_pred
#     #             _, pred_indices = test_outputs.max(dim=1)
#     #             _, target_indices = target.max(dim=1)
#     #             correct += (pred_indices == target_indices).sum().item()
#     #             total += target.shape[0]
#     #     summed_test_acc = correct/total

#         # # compute summed model output train acc
#         # total = 0
#         # correct = 0
#         # for data_idx, (data, target) in enumerate(train_loader):
#         #         data, target = (data.cuda(), target.cuda())
#         #         if k_classes == 10:
#         #             data, target = normalize_data_10_class(data, target)
#         #         else:
#         #             # normalize_data trains even/odd binary 
#         #             data, target = normalize_data(data, target)
#         #         # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#         #             if k_classes == 2:
#         #                 target = (target + 1).true_divide(2)
#         #         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#         #         target = target.cuda()
#         #         for idx, cur_model in enumerate(models_to_sum):
#         #             cur_model.cuda()
#         #             y_pred = cur_model(data).squeeze()
#         #             if idx == 0:
#         #                 train_outputs = y_pred
#         #             else:
#         #                 train_outputs += y_pred
#         #         _, pred_indices = train_outputs.max(dim=1)
#         #         _, target_indices = target.max(dim=1)
#         #         correct += (pred_indices == target_indices).sum().item()
#         #         total += target.shape[0]
#         # summed_train_acc = correct/total
#         exp_dict['summed_train_acc'] = summed_train_acc
#         exp_dict['summed_test_acc'] = summed_test_acc
#         all_exp_dicts.append(exp_dict)


# final_vals_dict = {'true_data_results': all_exp_dicts}  
# # if to_train_rand_data:
# #     final_vals_dict['rand_data_results'] = rand_data_results

# final_vals_dict['parameters'] = params_set

# f = open("ensemble_troubleshoot.pkl", "wb")
# pickle.dump(final_vals_dict, f)
# f.close()
#BLOCK HERE

""" Experiments examining the effect of empirical margin on network averaging
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
from torch.optim.optimizer import Optimizer
from fromage import Fromage
from util.nero import Nero
from util.data import get_data, normalize_data, normalize_data_10_class, get_data_k_class
from util.trainer import SimpleNet, SimpleNetMultiClass, train_network, train_network_multiclass, train_network_multiclass_combined, train_network_multiclass_scale_label
from util.trainer import train_network_multiclass_scale_label_input_net, generalized_multiclass_train, generalized_multiclass_train_fullbatch
from util.hessians import *
from util.scale_init import NetScaleInit
from matplotlib import cm
from generalization_bounds import *
#TODO: This code does not work with training networks on randomly shuffled data

# scale_vector = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 10, 20]
scale_vector = np.logspace(-2, 1, num=15)
# scale_vector.reverse()

num_networks = 15
epochs = 500 # 300
# lr_decay = 0.99
lr_decay = 0.992
lr  = 0.01
depth = 5
width = 2048
k_classes = 10
num_train_examples = 1000

tqdm_flag = True
tqdm_ = lambda x: x
cur_opt = Nero
# cur_opt = SGD
optimizer_kwargs = {'lr':lr, 'beta':0.999, 'constraints':True}
early_stop = False
to_spect_norm = False
# to_spect_norm = True

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

to_train_rand_data = False


delta = 0.05
# gamma_array = [10**gamma_mod for gamma_mod in range(0,11,2)]
params_set = {'num_networks': num_networks,
              'epochs': epochs,
              'lr': lr,
              'depth': depth,
              'width': width,
              'num_train_examples': num_train_examples,
              'lr_decay': lr_decay,
            #   'bartlett_gamma_array': gamma_array,
              'delta': delta,
              'optimizer': cur_opt,
            #   'true_data_label_scale': true_data_label_scale,
            #   'rand_label_scale': rand_label_scale,
            #   'sigma_vector': sigma_vector,
              'loss_function': criterion,
              'scale_vector': scale_vector
              }


all_exp_dicts = []
for exp_idx, cur_scale in tqdm(enumerate(scale_vector)):
    # Train all networks in an experiment on the same data
    # if exp_idx >0:
    #     torch.cuda.empty_cache()

    full_batch_train_loader, train_loader, test_loader = get_data(
        num_train_examples=num_train_examples,
        batch_size=num_train_examples,
        random_labels=False,
        binary_digits=False)

    models_to_sum = []
    models_dicts = []
    # exp_dict = {'models_dicts': models_dicts,
    #         'label_scale': cur_scale,
    #         'summed_train_acc':[],
    #         'summed_test_acc': []
    #         }
    for net in tqdm(range(num_networks)):

        true_data_results = {'train_acc_list': [],
                     'test_acc_list': [],
                     'correct_class_outputs': [],
                     'other_class_outputs': [],
                     'fro_norms': [],
                     'spect_norms': [],
                     'bartlett_spect_complexity': [],
                     'X_norms': []
        }

        true_data_label_scale = cur_scale
        rand_data_label_scale = cur_scale

        # model = NetScaleInit(depth, width, k_classes, sigma)
        model = SimpleNetMultiClass(depth, width, k_classes)
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
                                         early_stop=early_stop)
        
        fro_norm = []
        spectral_norm = []
        model.cpu()
    
        for idx, params in enumerate(model.parameters()):

            p = params
            p = p.detach()
            fro_norm.append(np.linalg.norm(p, ord='fro'))
            spectral_norm.append(np.linalg.norm(p, ord=2))
        spect_complex = bartlett_spectral_complexity(model, ref_M=init_weights)



        true_data_results['train_acc_list'].append(train_acc)
        true_data_results['test_acc_list'].append(test_acc)
        true_data_results['fro_norms'].append(fro_norm)
        true_data_results['spect_norms'].append(spectral_norm)
        true_data_results['correct_class_outputs'].append(correct_class_outputs)
        true_data_results['other_class_outputs'].append(other_class_outputs)
        true_data_results['bartlett_spect_complexity'].append(spect_complex)
        true_data_results['X_norms'].append(X_norm)
        # true_data_results['sigma'].append(sigma)

        models_dicts.append(true_data_results)
        models_to_sum.append(model)

    exp_dict = {'models_dicts': models_dicts,
                'label_scale': cur_scale,
                'summed_train_acc':[],
                'summed_test_acc': []
                }
        # unindent here          
    with torch.no_grad():
        # compute summed model output test acc
        total = 0
        correct = 0
        for data_idx, (data, target) in enumerate(test_loader):
                data, target = (data.cuda(), target.cuda())
                if k_classes == 10:
                    data, target = normalize_data_10_class(data, target)
                else:
                    # normalize_data trains even/odd binary 
                    data, target = normalize_data(data, target)
                # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
                    if k_classes == 2:
                        target = (target + 1).true_divide(2)

                target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
                target = target.cuda()
                for idx, cur_model in enumerate(models_to_sum):
                    cur_model.cuda()
                    y_pred = cur_model(data).squeeze()
                    if idx == 0:
                        test_outputs = y_pred
                    else:
                        test_outputs += y_pred
                _, pred_indices = test_outputs.max(dim=1)
                _, target_indices = target.max(dim=1)
                correct += (pred_indices == target_indices).sum().item()
                total += target.shape[0]
        summed_test_acc = correct/total

        # compute summed model output train acc
        total = 0
        correct = 0
        for data_idx, (data, target) in enumerate(train_loader):
                data, target = (data.cuda(), target.cuda())
                if k_classes == 10:
                    data, target = normalize_data_10_class(data, target)
                else:
                    # normalize_data trains even/odd binary 
                    data, target = normalize_data(data, target)
                # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
                    if k_classes == 2:
                        target = (target + 1).true_divide(2)
                target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
                target = target.cuda()
                for idx, cur_model in enumerate(models_to_sum):
                    cur_model.cuda()
                    y_pred = cur_model(data).squeeze()
                    if idx == 0:
                        train_outputs = y_pred
                    else:
                        train_outputs += y_pred
                _, pred_indices = train_outputs.max(dim=1)
                _, target_indices = target.max(dim=1)
                correct += (pred_indices == target_indices).sum().item()
                total += target.shape[0]
        summed_train_acc = correct/total
        exp_dict['summed_train_acc'] = summed_train_acc
        exp_dict['summed_test_acc'] = summed_test_acc
        all_exp_dicts.append(exp_dict)


final_vals_dict = {'true_data_results': all_exp_dicts}  
# if to_train_rand_data:
#     final_vals_dict['rand_data_results'] = rand_data_results

final_vals_dict['parameters'] = params_set

f = open(f"faster_ensemble_networks_more_granular_nero_{depth}depth_{width}width_{num_networks}nets_{epochs}epochs.pkl", "wb")
pickle.dump(final_vals_dict, f)
f.close()
