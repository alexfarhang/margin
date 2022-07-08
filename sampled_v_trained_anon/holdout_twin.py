import math
import torch
import pickle
import itertools
import numpy as np
import copy
from tqdm import tqdm

from absl import flags
from absl import app
import json

import csv
import os

FLAGS = flags.FLAGS

# from util.cifar10_data import get_data, normalize_data_even_odd
from util.trainer import AdaptiveConvNet, ConvNet, SimpleNet
from util.nero import Nero,neuron_mean,neuron_norm

import time

import torch.optim 
import gc

# Define flags
flags.DEFINE_bool('sampled_constraints', True, 'Whether the sampled networks should have weight constraints')
flags.DEFINE_bool('trained_constraints', True, 'Whether the trained networks should have weight constraints')
flags.DEFINE_float('loss_threshold', 0.01, 'The loss threshold to train the networks until i.e (train_network_output - sampled_network_output) < eps')
flags.DEFINE_integer('num_perfect_models', 100, 'Number of sampled networks that perfectly fit training data to find')
flags.DEFINE_integer('num_train_epochs', 1000, 'The number of epochs to train networks to match sampled network output')
flags.DEFINE_integer('num_train_examples', 5, 'The number of training datapoints to consider')
flags.DEFINE_integer('num_test_examples', 50, 'The number of test datapoints to consider')
flags.DEFINE_bool('random_labels', False, 'Whether or not to assign random labels to training data')
flags.DEFINE_bool('binary_digits', False, 'Whether or not to turn MNIST data into binary 0 or 1 classification')
flags.DEFINE_integer('depth', 7, 'Depth of model architecture to consider')
flags.DEFINE_integer('width', 7000, 'Width of model architecture to consider')
flags.DEFINE_bool('lr_scheduling', False, 'Whether or not to turn on LR decay for optimizer')
flags.DEFINE_string('optimizer', 'Nero', 'Optimizer to train networks with')
flags.DEFINE_float('init_lr', 0.001, 'Initial LR to train networks with (might go through LR decay depending on flag lr_schedling)')
flags.DEFINE_float('decay', 0.9, 'Constant by which to decay LR when training networks (if lr_scheduling flags is True)')
flags.DEFINE_integer('seed', 0, 'Seed to for numpy/torch (affects which training/test examples the experiment is run on)')
flags.DEFINE_integer('iter_until_exit', 4000, 'Number of iterations to run code until we sample num_perfect_models that perfectly fit training data')
flags.DEFINE_string('mode', 'conditional_sampling_seed', 'Which mode to run the sampling in (joint_sampling_same_test, joint_sampling_diff_test, conditional_sampling_seed)')
flags.DEFINE_bool('norm_tracking', True, 'Whether or not to output frob/spectral norm of the sampled/trained networks')
flags.DEFINE_integer('robustness_iter_count', 5, 'How many iterations to repeat robustness measure over')
flags.DEFINE_float('eta', 0.01, 'Strength of robustness perturbation delta_w matrix')
flags.DEFINE_bool('save_models', False, 'Whether or not to save models')
flags.DEFINE_string('result_dir', '', 'Where to save models (if empty, will not save models)')
flags.DEFINE_bool('directly_compute_measures', False, 'Whether to directly compute the measures')
flags.DEFINE_string('network_arch', 'linear', 'whether or not to keep architect linear')
flags.DEFINE_string('dataset_type', 'mnist0v1', 'dataset type')
flags.DEFINE_integer('conv_depth', -1, 'when using network_arch=conv_adaptive, the depth of conv layers to use')
flags.DEFINE_integer('conv_width', -1, 'when using network_arch=conv_adaptive, the width of conv layers to use')
flags.DEFINE_integer('mnist_first_class', -1, 'first class to use with mnist binary classification')
flags.DEFINE_integer('mnist_second_class', -1, 'second class to use with mnist binary classification')


### Data hyperparameters (note: doesn't matter if num_train_examples < batch_size)
batch_size = 50

### Estimator hyperparameters
cuda = False

def compute_model_norms(model):

    # model sampled
    spec_norm_result_sampled = 0.0
    for w in model.parameters():
        if len(w.shape) == 2:
            spec_norm_cand = torch.linalg.norm(w, ord=2)

            if torch.absolute(spec_norm_cand) != 0:
                spec_norm_result_sampled += torch.log(spec_norm_cand)
    return spec_norm_result_sampled


def main(argv):
    # Retrieve flags
    sampled_constraints = FLAGS.sampled_constraints
    trained_constraints = FLAGS.trained_constraints
    loss_threshold = FLAGS.loss_threshold
    num_perfect_models = FLAGS.num_perfect_models
    num_train_epochs = FLAGS.num_train_epochs
    num_train_examples = FLAGS.num_train_examples
    num_test_examples = FLAGS.num_test_examples
    random_labels = FLAGS.random_labels
    binary_digits = FLAGS.binary_digits
    depth = FLAGS.depth
    width = FLAGS.width
    lr_scheduling = FLAGS.lr_scheduling
    optimizer = FLAGS.optimizer
    init_lr = FLAGS.init_lr
    decay = FLAGS.decay
    seed = FLAGS.seed
    iter_until_exit = FLAGS.iter_until_exit
    mode = FLAGS.mode
    result_dir = FLAGS.result_dir
    save_models = FLAGS.save_models
    directly_compute_measures = FLAGS.directly_compute_measures
    network_arch = FLAGS.network_arch
    dataset_type = FLAGS.dataset_type
    conv_depth = FLAGS.conv_depth
    conv_width = FLAGS.conv_width
    mnist_first_class = FLAGS.mnist_first_class
    mnist_second_class = FLAGS.mnist_second_class

    device_1 = torch.device("cpu")
    # device_1 = torch.device('cuda:0')
    # device_2 = torch.device('cuda:1')

    if network_arch == "conv_adaptive":
        if conv_width == -1 or conv_depth == -1:
            print("Please provide conv depth and width when using conv adaptive")
            return 

        print(f"conv width: {conv_width}")
        print(f"conv depth: {conv_depth}")
        print(f"lin width: {width}")
        print(f"lin depth: {depth}")

    # if save_models:
    result_csv_path = None
    fieldnames = [
        'seed_num', 
        'sampled_model_acc', 
        'trained_model_acc', 
        'good_fit', 
        'sampled_model_filepath', 
        'trained_model_filepath',
        'flip',
        'margin_sampled',
        'margin_trained',
        'error_trained',
        'spec_norm_sampled',
        'spec_norm_trained',
        'rel_err_samp',
        'rel_err_train'
    ]
    result_csv_filename = f"result_summary_seed_{seed}.csv"
    result_csv_path = os.path.join(os.getcwd(), result_dir, result_csv_filename)

    # Check that the save directory is not a bad arg
    # if result_dir != '' and (os.path.isdir(result_dir)):
    #         # save the sampled and trained model to a directory
    #         print("Exiting... you are about to override current results")
    #         return
    if result_dir != '':
        if (not os.path.isdir(result_dir)):
            print("Creating directory...")
            os.mkdir(result_dir)
        # Save metadata
        metadata = {}
        metadata["random_labels"] = random_labels
        metadata["binary_digits"] = binary_digits
        metadata["batch_size"] = batch_size
        metadata["num_train_examples"] = num_train_examples
        metadata["num_test_examples"] = num_test_examples
        metadata["mode"] = mode
        metadata["original_seed"] = seed
        metadata["depth"] = depth
        metadata["width"] = width

        with open(os.path.join(os.getcwd(), result_dir, f"result_metadata_seed_{seed}.txt"), "w+") as outfile:
            json.dump(metadata, outfile)
    
        with open(result_csv_path, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()


    # If directly_compute_measures is true, create the output measures file 


    measure_file_headers = \
        [
            'seed', 'sampled_model_acc', 'trained_model_acc', 'sam_model_acc', 'good_fit', 'sam_model_good_fit', 
            'flip', 'iter_based_seed', 'sampled_worst_case_norm_diff', 'sampled_worst_case_margin_diff', 
            'trained_worst_case_norm_diff', 'trained_worst_case_margin_diff', 'sam_worst_case_norm_diff', 
            'sam_worst_case_margin_diff', 'CT.L2_sampled', 'CT.PARAMS_sampled', 'CT.INVERSE_MARGIN_sampled', 
            'CT.LOG_PROD_OF_SPEC_sampled', 'CT.LOG_PROD_OF_SPEC_OVER_MARGIN_sampled', 'CT.FRO_OVER_SPEC_sampled', 
            'CT.LOG_SPEC_ORIG_MAIN_sampled', 'CT.LOG_SUM_OF_SPEC_OVER_MARGIN_sampled', 'CT.LOG_SUM_OF_SPEC_sampled', 
            'CT.LOG_PROD_OF_FRO_sampled', 'CT.LOG_PROD_OF_FRO_OVER_MARGIN_sampled', 
            'CT.LOG_SUM_OF_FRO_OVER_MARGIN_sampled', 'CT.LOG_SUM_OF_FRO_sampled', 'CT.PARAM_NORM_sampled', 
            'CT.PATH_NORM_sampled', 'CT.PATH_NORM_OVER_MARGIN_sampled', 
            'CT.PACBAYES_ORIG_sampled', 'CT.PACBAYES_FLATNESS_sampled', 'CT.PACBAYES_ALPHA_ORIG_sampled', 'CT.PACBAYES_ALPHA_FLATNESS_sampled', 
            'CT.L2_trained', 'CT.PARAMS_trained', 'CT.INVERSE_MARGIN_trained', 'CT.LOG_PROD_OF_SPEC_trained', 
            'CT.LOG_PROD_OF_SPEC_OVER_MARGIN_trained', 'CT.FRO_OVER_SPEC_trained', 'CT.LOG_SPEC_ORIG_MAIN_trained', 
            'CT.LOG_SUM_OF_SPEC_OVER_MARGIN_trained', 'CT.LOG_SUM_OF_SPEC_trained', 'CT.LOG_PROD_OF_FRO_trained', 
            'CT.LOG_PROD_OF_FRO_OVER_MARGIN_trained', 'CT.LOG_SUM_OF_FRO_OVER_MARGIN_trained', 
            'CT.LOG_SUM_OF_FRO_trained', 'CT.PARAM_NORM_trained', 'CT.PATH_NORM_trained', 
            'CT.PATH_NORM_OVER_MARGIN_trained', 
            'CT.PACBAYES_ORIG_trained', 'CT.PACBAYES_FLATNESS_trained', 'CT.PACBAYES_ALPHA_ORIG_trained', 'CT.PACBAYES_ALPHA_FLATNESS_trained', 
            'trained_rand_dir_std', 'sampled_rand_dir_std', 'trained_rand_dir_mean', 'sampled_rand_dir_mean'
        ]

    if directly_compute_measures:

        with open(f"{result_dir}/full_measures_{seed}.csv", "w") as measure_result_file:
            writer = csv.DictWriter(measure_result_file, fieldnames=measure_file_headers)
            writer.writeheader()


    # Seeding torch/numpy
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # if dataset_type == "mnist":
    #     from util.data import get_data, normalize_data_even_odd
    if dataset_type == "mnist":
        assert(mnist_first_class != -1)
        assert(mnist_second_class != -1)
        from util.mnist3v8 import get_data, normalize_data_even_odd
    elif dataset_type == "cifar10":
        if network_arch == "conv" or network_arch == 'conv_adaptive':
            from util.cifar10_data_unflat import get_data, normalize_data_even_odd
        elif network_arch == "linear":
            from util.cifar10_data import get_data, normalize_data_even_odd
        else:
            print("not a valid architecter...exiting..")
            return
    else:
        print('uhoh2')
        return
    # Retrieve the training/test data
    data = get_data( num_train_examples=num_train_examples,
                        num_test_examples=num_test_examples,
                        batch_size=batch_size, 
                        random_labels=random_labels, 
                        binary_digits=binary_digits,
                        mnist_first_class=mnist_first_class,
                        mnist_second_class=mnist_second_class
                    )

    # Get the train loaders
    full_batch_train_loader, full_batch_test_loader, _, _ = data

    train_acc_list = []
    test_acc_list_sampled = []
    test_acc_list_trained = []
    test_good_fit_list = []

    test_acc_list_sam = []
    test_good_fit_list_sam = []

    num_perfect_models_found = 0

    for seed_num in tqdm(range(iter_until_exit)):
        torch.cuda.empty_cache()
        # Choose training/test data based on mode
        if mode == "joint_sampling_same_test":
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

            data = get_data( num_train_examples=num_train_examples,
                        num_test_examples=num_test_examples,
                        batch_size=batch_size, 
                        random_labels=random_labels, 
                        binary_digits=binary_digits )

            full_batch_train_loader, _, _, _ = data
        elif mode == "joint_sampling_diff_test":
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

            data = get_data( num_train_examples=num_train_examples,
                        num_test_examples=num_test_examples,
                        batch_size=batch_size, 
                        random_labels=random_labels, 
                        binary_digits=binary_digits )

            full_batch_train_loader, full_batch_test_loader, _, _ = data   

        # If we've found num_perfect_models sampled networks that fit training data, exit
        if num_perfect_models_found >= num_perfect_models:
            break

        # Load the data and targets ONE TIME
        train_data, train_target = list(full_batch_train_loader)[0]
        test_data, test_target = list(full_batch_test_loader)[0]

        if cuda:
            train_data, train_target = (train_data.cuda(), train_target.cuda())
            test_data, test_target = (test_data.cuda(), test_target.cuda())

        train_data, train_target = normalize_data_even_odd(train_data, train_target)
        test_data, test_target = normalize_data_even_odd(test_data, test_target)
        
        # Sampled network code
        model = None
        if network_arch == 'linear':
            initial_dim = 784 if "mnist" in dataset_type else 3072
            model = SimpleNet(initial_dim, depth, width)
        elif network_arch == 'conv_adaptive':
            model = AdaptiveConvNet(conv_depth, conv_width, depth, width)
        elif network_arch == 'conv':
            model = ConvNet()
        else:
            print('no valid architect...exiting..')
            return
        
        if cuda:
            model = model.to(device_1)
            train_data = train_data.to(device_1)
            train_target = train_target.to(device_1)
            test_data = test_data.to(device_1)
            test_target = test_target.to(device_1)

        # Constrain the sampled parameters if sampled_constraints = True
        for p in model.parameters():
            p.data = torch.randn_like(p) / math.sqrt(p.shape[1])
            if sampled_constraints and p.dim() > 1:
                p.data -= neuron_mean(p.data)
                p.data /= neuron_norm(p.data)

        # Evaluate model on test set
        model.eval()
        correct = 0
        total = 0

        y_pred = model(train_data).squeeze()

        correct += (train_target.float() == y_pred.sign()).sum().item()
        total += train_target.shape[0]

        train_acc = correct/total
        y_target = y_pred.clone() * 1.0
        
        # If we perfectly classify or misclassify the data
        if (train_acc == 1.0 or train_acc == 0.0):

            t0 = time.time()
            
            if train_acc == 0.0:
                print("flipped")
                flip = -1
            else:
                flip = 1

            num_perfect_models_found += 1

            y_pred_sampled = model(test_data).squeeze() * flip
            y_pred_sampled_san = model(train_data).squeeze() * flip

            # train a model to get the same output as the sampled model
            model_trained = None
            if network_arch == 'linear':
                initial_dim = 784 if "mnist" in dataset_type else 3072
                model_trained = SimpleNet(initial_dim, depth, width)
            elif network_arch == 'conv_adaptive':
                model_trained = AdaptiveConvNet(conv_depth, conv_width, depth, width)
            elif network_arch == 'conv':
                model_trained = ConvNet()
            else:
                print("Not a valid arch...exiting..")
                return
            
            if cuda:
                model_trained = model_trained.to(device_1)
                train_data = train_data.to(device_1)
                train_target = train_target.to(device_1)
                test_data = test_data.to(device_1)
                test_target = test_target.to(device_1)
                y_target = y_target.to(device_1)

            for p in model_trained.parameters():
                p.data = torch.randn_like(p) / math.sqrt(p.shape[1])
                if sampled_constraints and p.dim() > 1:
                    p.data -= neuron_mean(p.data)
                    p.data /= neuron_norm(p.data)
                    
            if optimizer == 'Nero':
                optim = Nero(model_trained.parameters(), lr=init_lr,constraints=trained_constraints)
            elif optimizer == 'SGD':
                optim = torch.optim.SGD(model_trained.parameters(), lr=init_lr)
            elif optimizer == 'Adam':
                optim = torch.optim.Adam(model_trained.parameters(), lr=init_lr)
            else:
                print("Please choose a valid optimizer!")
                return

            if lr_scheduling:
                lr_lambda = lambda x: decay**x
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        
            good_fit = False

            for epoch in tqdm(range(num_train_epochs)):

                y_pred_train = model_trained(train_data).squeeze()

                loss = (y_pred_train - y_target).norm()

                if loss < loss_threshold:
                    good_fit = True
                    break

                model_trained.zero_grad()    
                loss.backward(retain_graph=True)
                optim.step()
                
                if lr_scheduling:
                    lr_scheduler.step()
            if not good_fit:
                print("Regression threshold not reached!!!")
            print("target output: ", train_target)
            print("trained output: ", y_pred)
            print("sampled output: ", y_target)
            print("epoch: ", epoch,"loss: ", loss.mean())



            correct_sampled = 0
            total_sampled = 0
            correct_trained = 0
            total_trained = 0

                
            # y_pred_sampled = model(test_data).squeeze() * flip
            y_pred_trained = model_trained(test_data).squeeze() * flip
            y_pred_trained_san = model_trained(train_data).squeeze() * flip


      
            correct_sampled += (test_target.float() == y_pred_sampled.sign()).sum().item()
            correct_trained += (test_target.float() == y_pred_trained.sign()).sum().item()
            total_sampled += test_target.shape[0]
            total_trained += test_target.shape[0]

   
            test_acc_sampled = correct_sampled/total_sampled
            print("test accuracy of sampled model: ", test_acc_sampled)

            test_acc_trained = correct_trained/total_trained
            print("test accuracy of trained model: ", test_acc_trained)

            train_acc_list.append(train_acc)
            test_acc_list_sampled.append(test_acc_sampled)
            test_acc_list_trained.append(test_acc_trained)

            good_fit_num = 1 if good_fit else 0
            test_good_fit_list.append(good_fit_num)

            sampled_model_filepath = None
            trained_model_filepath = None

            # delete unnecessary models

            # Saving the model
            if save_models:
                # save the sampled and trained model to a directory
                sampled_model_filepath = os.path.join(os.getcwd(), result_dir, f"model_sampled_seed_{seed_num}.pt")
                trained_model_filepath = os.path.join(os.getcwd(), result_dir, f"model_trained_seed_{seed_num}.pt")
                torch.save(model.state_dict(), sampled_model_filepath)
                torch.save(model_trained.state_dict(), trained_model_filepath)

            true_seed = seed if mode == "conditional_sampling" else seed_num
        
            # Write the row in the CSV
            # if save_models:
            try:
                # spec_norm_sampled = compute_model_norms(model).item()
                # spec_norm_trained = compute_model_norms(model_trained).item()
                spec_norm_sampled = -1
                spec_norm_trained = -1
            except Exception:
                spec_norm_sampled = -1
                spec_norm_trained = -1

            # if directly_compute_measures:
            with open(result_csv_path, mode='a') as csv_file:
                marg_samp = torch.mean(torch.absolute(y_pred_sampled_san)).item()
                marg_train = torch.mean(torch.absolute(y_pred_trained_san)).item()
                final_loss = (model_trained(train_data).squeeze() - y_target).norm().item()

                rel_err_samp = final_loss / marg_samp
                rel_err_train = final_loss / marg_train

                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                
                writer.writerow({
                    'seed_num': true_seed,
                    'sampled_model_acc': test_acc_sampled,
                    'trained_model_acc': test_acc_trained,
                    'good_fit': str(good_fit_num),
                    'sampled_model_filepath': sampled_model_filepath,
                    'trained_model_filepath': trained_model_filepath,
                    'flip': flip,
                    'margin_sampled': 1/(marg_samp ** 2),
                    'margin_trained': 1/(marg_train ** 2),
                    'error_trained': final_loss,
                    'spec_norm_sampled': spec_norm_sampled,
                    'spec_norm_trained': spec_norm_trained,
                    "rel_err_samp": rel_err_samp,
                    "rel_err_train": rel_err_train
                })


            # Running measures (if directly_compute_measures is True)
            if directly_compute_measures:

                row = {}
                row["seed"] = seed
                row["sampled_model_acc"] = test_acc_sampled
                row["trained_model_acc"] = test_acc_trained
                row["good_fit"] = good_fit_num
                row["flip"] = flip
                row["iter_based_seed"] = seed_num

                # ensure the train accuraciea are 100%
                # y_pred_sampled_san = model(train_data).squeeze() * flip
                # y_pred_trained_san = model_trained(train_data).squeeze() * flip
                correct_sampled = 0
                total_sampled = 0
                correct_trained = 0
                total_trained = 0

                correct_sampled += (train_target.float() == y_pred_sampled_san.sign()).sum().item()
                correct_trained += (train_target.float() == y_pred_trained_san.sign()).sum().item()
                total_sampled += train_target.shape[0]
                total_trained += train_target.shape[0]

                train_acc_sampled = correct_sampled/total_sampled

                train_acc_trained = correct_trained/total_trained
                
                if train_acc_sampled != 1.0 or train_acc_trained != 1.0:
                    print("Saved models do not reach 100% training accuracy on training dataset")
                    print(f"train acc sampled: {train_acc_sampled}")
                    print(f"train acc trained nero: {train_acc_trained}")
                    num_perfect_models_found -= 1
                else:
                    # model = model.to(device_2)
                    # model_trained = model_trained.to(device_2)
                    # train_data = train_data.to(device_2)
                    # train_target = train_target.to(device_2)
                    # test_data = test_data.to(device_2)
                    # test_target = test_target.to(device_2)
                    # y_target = y_target.to(device_2)
                    # try:
                    our_measures_row = \
                        measures.compute_measures_one_row(
                            model, 
                            model_trained,
                            train_data, 
                            train_target,
                        )

                    # Compute outside measures
                    outside_measures_row_sampled = get_all_measures(
                        model=model,
                        dataloader=full_batch_train_loader,
                        acc=train_acc_sampled,
                        seed=seed,
                        flip=flip
                    )

                    new_row_sampled = {str(key) + "_sampled": value for key, value in outside_measures_row_sampled.items()}

                    outside_measures_row_trained = get_all_measures(
                        model=model_trained,
                        dataloader=full_batch_train_loader,
                        acc=train_acc_trained,
                        seed=seed,
                        flip=flip
                    )

                    new_row_trained = {str(key) + "_trained": value for key, value in outside_measures_row_trained.items()}


                    t1 = time.time()
                    time_elapsed = t1 - t0
                    print(f"time elapsed is: {time_elapsed}")

                    final_row = {
                        **row, 
                        **our_measures_row, 
                        **new_row_sampled,
                        **new_row_trained,
                    }
                
                    # save all the results to a csv file
                    with open(f"{result_dir}/full_measures_{seed}.csv", mode='a') as measure_result_file:
                        writer = csv.DictWriter(measure_result_file, fieldnames=measure_file_headers)
                        writer.writerow(final_row)

                    # except Exception as e:
                    #     print(f"failed to compute measures on on seed: {seed_num}")
                    #     print(e)

    print("sampled constraints:", sampled_constraints, "trained constraints: ", trained_constraints)
    print("Test accuracies of sampled networks")
    for i in test_acc_list_sampled:
        print(i)

    print("Test accuracies of trained networks")
    for i in test_acc_list_trained:
        print(i)

    print("Test good fit list")
    for i in test_good_fit_list:
        print(i)

    print("Test accuracies of SAM-trained networks")
    for i in test_acc_list_sam:
        print(i)

    print("Test good fit list of SAM-trained networks")
    for i in test_good_fit_list_sam:
        print(i)

if __name__ == "__main__":
    app.run(main)