import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as st
import torch
from matplotlib import cm



def get_margin_one_network(data_result, metric, other_metric, net_num, num_train_examples, to_normalize):
    """Calculate margins for a single network (with multiclass outputs).  
    Margins are in the form of F_A(x)_y - max_{i!=y}F_A(x)
    Normalized are divided by (R_A * ||X||_2 / n)

    Inputs:
    data_result: data dictionary with key [metric]
    metric: FA(x)_i.  Choose correct class and maximal other class
    other_metric: maximal incorrect label activation
    net_num: index data_results by the network of interest
    to_normalize: True for Bartlett spectral complexity normalization

    Example:
    true_margins_fr = get_margin_one_network(true_data_results_fr, \
    'correct_class_outputs', 'other_class_outputs', net_num, \
    to_normalize=False).numpy()
    """
    margins = data_result[metric][net_num] - data_result[other_metric][net_num]
    if to_normalize:
        X_norm = data_result['X_norms'][net_num]
        margins /= data_result['bartlett_spect_complexity'][net_num] * X_norm * (1/num_train_examples)
    return margins


def plot_n_network_margins_and_normalized_margins(true_data_results,
                                                  rand_data_results,
                                                  num_training_examples,
                                                  n=10,
                                                  metric1='correct_class_outputs',
                                                  metric2='other_class_outputs',
                                                  param_vector=False,
                                                  param_string=False,
                                                  bins=30):
    """Create plot of n rows (from different networks) of margins and 
    two columns (absolute margins and spectral complexity normalized ones)

    n should equal len(param_vector)
    
    """
    if not param_vector:
        param_vector = np.arange(n)
    if not param_string:
        param_string = 'net'

    indices = [x for x in range(1,n*2+1)]
    counter = 0
    for net_num in range(n):
        plt.subplot(n, 2, indices[counter])
        if not net_num:
            plt.title('Margins')

        true_margins = get_margin_one_network(true_data_results, 'correct_class_outputs', 'other_class_outputs', net_num, num_training_examples, to_normalize=False).numpy()
        rand_margins = get_margin_one_network(rand_data_results, 'correct_class_outputs', 'other_class_outputs', net_num, num_training_examples, to_normalize=False).numpy()

        true_margins_normalized = get_margin_one_network(true_data_results, 'correct_class_outputs', 'other_class_outputs', net_num, num_training_examples, to_normalize=True).numpy()
        rand_margins_normalized = get_margin_one_network(rand_data_results, 'correct_class_outputs', 'other_class_outputs', net_num, num_training_examples, to_normalize=True).numpy()
        plt.hist(true_margins, alpha=0.8, label='true', bins=bins)
        plt.hist(rand_margins, alpha=0.8, label='rand', bins=bins)
        if net_num == 0:
            # constrain all x axes to the same interval
            ax1 = plt.gca()
            x_lim1 = ax1.get_xlim()
        else:
            plt.xlim(x_lim1)
        if net_num == n-1:
            plt.xlabel('unnormalized margin: (f(x_i)yi - max(other))')
        plt.ylabel(f"{param_string}: {np.round(param_vector[net_num],2)}")
        counter += 1

        # xlims = plt.xlim()
        plt.subplot(n, 2, indices[counter])
        if not net_num:
            plt.title('Spectrally normalized margins')
        plt.hist(true_margins_normalized, alpha=0.8, label='true_normalized', bins=bins)
        plt.hist(rand_margins_normalized, alpha=0.8, label='rand_normalized', bins=bins)
    #     plt.legend()
        # plt.xlim(xlims)
        if net_num == 0:
            ax2 = plt.gca()
            x_lim2 = ax2.get_xlim()
        else:
            plt.xlim(x_lim2)
            
        counter += 1
        if net_num == len(param_vector)-1:
            plt.legend()
            plt.xlabel('normalized margin: (f(x_i)yi - max(other))/spectral complexity')
    fig = plt.gcf()
    fig.set_size_inches(24, 12)


# Ensembles and margins
def extract_ensemble_performance(data_results, data_list_str, parameters):
    '''Calculate the mean, min, max performance of the ensembles for data_results'''
    bb = []
    # j for each experiment (scale_vector)
    for j in range(len(parameters['scale_vector'])):
        # i for each network in the experiment
        b = np.array([data_results[j]['models_dicts'][i][data_list_str][0][-1] for i in range(parameters['num_networks'])])
#         print(b)
        bb.append(b)

    bb = np.array(bb)
    return bb.mean(axis=1), bb.min(axis=1), bb.max(axis=1)


def plot_ensemble_performance(data_results, parameters, c0, c1):
    scale_vector = parameters['scale_vector']
    num_networks = parameters['num_networks']
    alpha=0.5
    
    mean_net_performance_train, min_net_performance_train, max_net_performance_train = extract_ensemble_performance(
        data_results, 'train_acc_list', parameters)
    mean_net_performance_test, min_net_performance_test, max_net_performance_test = extract_ensemble_performance(
        data_results, 'test_acc_list', parameters)
        
    # errors must be in the form [2,N] lower then upper
    lower_error_train = np.abs(min_net_performance_train - mean_net_performance_train).reshape(1, len(scale_vector))
    lower_error_test = np.abs(min_net_performance_test - mean_net_performance_test).reshape(1, len(scale_vector))
    upper_error_train = (max_net_performance_train - mean_net_performance_train).reshape(1, len(scale_vector))
    upper_error_test = (max_net_performance_test - mean_net_performance_test).reshape(1, len(scale_vector))
    combo_error_train = np.concatenate((lower_error_train, upper_error_train), axis=0)
    combo_error_test = np.concatenate((lower_error_test, upper_error_test), axis=0)

    plt.scatter(scale_vector, mean_net_performance_train, label=f'train_{num_networks}_networks', alpha=alpha, color=c0, edgecolors='k', s=100)
    plt.scatter(scale_vector, mean_net_performance_test, label=f'test_{num_networks}_networks', alpha=alpha, color=c1, edgecolors='k', s=100)
    plt.errorbar(x=scale_vector, y=mean_net_performance_train, yerr=combo_error_train) #, fmt='none')
    plt.errorbar(x=scale_vector, y=mean_net_performance_test, yerr=combo_error_test)#, fmt='none')
    
    train_ensembled = np.array([data_results[i]['summed_train_acc'] for i in range(len(data_results))])
    test_ensembled = np.array([data_results[i]['summed_test_acc'] for i in range(len(data_results))])
    plt.scatter(scale_vector, train_ensembled, marker='s', label=f'{num_networks}_ensembled_train', alpha=alpha, color=c0, edgecolors='k', s=100)
    plt.scatter(scale_vector, test_ensembled, marker='s', label=f'{num_networks}_ensembled_test', alpha=alpha, color=c1, edgecolors='k', s=100)


    plt.xlabel('targeted margin')
    plt.ylabel('accuracy')
    plt.title(f"Ensembled vs mean network performance for ensembles of ({parameters['depth']}-layer nets)")
    plt.xscale('log')



    # plt.legend()
    # fig = plt.gcf()
    # fig.set_size_inches(12, 8)


def load_ensemble_data_and_plot(fname_list):
    """Load pickle file"""
    c0_map = cm.get_cmap('winter', len(fname_list))
    c1_map = cm.get_cmap('hot', len(fname_list))
    for idx, fname in enumerate(fname_list):
        f = open(fname, 'rb')
        results_dict_ensemble = pickle.load(f)
        f.close()
        parameters_ensemble = results_dict_ensemble['parameters']
        true_data_results_ensemble = results_dict_ensemble['true_data_results']

        plot_ensemble_performance(true_data_results_ensemble, parameters_ensemble, c0_map(idx), c1_map(idx))
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 8)


        


