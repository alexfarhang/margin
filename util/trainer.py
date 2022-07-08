import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.nero import Nero
from util.data import normalize_data, normalize_data_10_class

from generalization_bounds import bartlett_X_norm, collect_parameters_multiclass

from tqdm import tqdm
# tqdm = lambda x: x


class SimpleNet(nn.Module):
    def __init__(self, depth, width):
        super(SimpleNet, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, 1, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)

class SimpleNetMultiClass(nn.Module):
    def __init__(self, depth, width, k_classes):
        super(SimpleNetMultiClass, self).__init__()

        self.initial = nn.Linear(784, width, bias=False)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth-2)])
        self.final = nn.Linear(width, k_classes, bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)


# def train_network(train_loader, test_loader, depth, width, epochs, init_lr, decay, tqdm_flag=False, return_init=False, return_margins=False):    
    
#     if not tqdm_flag:
#         tqdm = lambda x: x
#     else:
#         from tqdm import tqdm

#     tqdm_  = lambda x: x
#     model = SimpleNet(depth, width).cuda()
#     optim = Nero(model.parameters(), lr=init_lr)      
#     lr_lambda = lambda x: decay**x
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    
#     if return_init:
#         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

#     model.train()
    
#     train_acc_list = []

#     for epoch in tqdm(range(epochs)):

#         correct = 0
#         total = 0

#         for data, target in tqdm_(train_loader):
#             data, target = (data.cuda(), target.cuda())
#             # normalize_data trains even/odd binary 
#             data, target = normalize_data(data, target)

#             y_pred = model(data).squeeze()
#             y_pred_train = y_pred.detach().cpu()
#             loss = (y_pred - target).norm()

#             correct += (target.float() == y_pred.sign()).sum().item()
#             total += target.shape[0]

#             model.zero_grad()
#             loss.backward()
#             optim.step()

#         lr_scheduler.step()
#         train_acc_list.append(correct/total)

#     model.eval()
#     correct = 0
#     total = 0 

#     for data, target in tqdm_(test_loader):
#         data, target = (data.cuda(), target.cuda())
#         data, target = normalize_data(data, target)
        
#         y_pred = model(data).squeeze()
#         correct += (target.float() == y_pred.sign()).sum().item()
#         total += target.shape[0]

#     if return_margins:
#         margins = y_pred_train
#     else:
#         margins = None

#     test_acc = correct/total
#     if return_init:
#         return train_acc_list, test_acc, model, init_weights, margins
#     else:
#         return train_acc_list, test_acc, model

# def train_network_multiclass(train_loader, test_loader, depth, width, k_classes, optimizer, epochs, init_lr, decay, tqdm_flag=False, return_init=True, return_margins=False):
#     """ multiclass version of train_network"""
#     # train_loader = full_batch_train_loader
#     # init_lr = lr
#     # decay = lr_decay
#     # tqdm_flag = False
#     # return_init = True
#     # return_margins = True
#     criterion = nn.MSELoss()
#     if not tqdm_flag:
#         tqdm = lambda x: x
#     else:
#         from tqdm import tqdm
#     tqdm_  = lambda x: x
#     model = SimpleNetMultiClass(depth, width, k_classes).cuda()
#     # optim = Nero(model.parameters(), lr=init_lr)
#     # optim = SGD(model.parameters(), lr=init_lr)      
#     optim = optimizer(model.parameters(), lr=init_lr)
#     lr_lambda = lambda x: decay**x
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

#     if return_init:
#         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

#     model.train()

#     train_acc_list = []

#     for epoch in tqdm(range(epochs)):

#         correct = 0 
#         total = 0

#         for data, target in tqdm_(train_loader):
#             data, target = (data.cuda(), target.cuda())
#             # normalize_data trains even/odd binary 
#             data, target = normalize_data(data, target)

#             y_pred = model(data).squeeze()
#             y_pred_train = y_pred.detach().cpu()

#             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#             target = (target + 1).true_divide(2)
#             target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#             target = target.cuda()

#             # loss = (y_pred - target).norm()
#             loss = criterion(y_pred.float(), target.float())
#             _, pred_indices = y_pred.max(dim=1)
#             _, target_indices = target.max(dim=1)
#             correct += (pred_indices == target_indices).sum().item()
#             # correct += (target.float() == y_pred.float().sign()).sum().item()
#             total += target.shape[0]

#             if epoch == epochs - 1:
#                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
#                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#                 other_class_outputs = other_class_values
#                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#                 true_train_targets = target

#             model.zero_grad()
#             loss.backward()
#             optim.step()

#         lr_scheduler.step()
#         train_acc_list.append(correct/total)

#     model.eval()
#     correct = 0
#     total = 0 

#     for data, target in tqdm_(test_loader):
#         data, target = (data.cuda(), target.cuda())
#         data, target = normalize_data(data, target)
        
#         y_pred = model(data).squeeze()
#         # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#         target = (target + 1).true_divide(2)
#         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#         target = target.cuda()

#         _, pred_indices = y_pred.max(dim=1)
#         _, target_indices = target.max(dim=1)
#         correct += (pred_indices == target_indices).sum().item()

#         # correct += (target.float() == y_pred.float().sign()).sum().item()
#         total += target.shape[0]

#     if return_margins:
#         margins = y_pred_train
#     else:
#         margins = None

#     test_acc = correct/total
    
#     # Compute X_norm:
#     X, Y, W = collect_parameters_multiclass(model, train_loader)
#     n = X.shape[0]

#     reshaped_X = X
#     X_norm = bartlett_X_norm(reshaped_X)

#     if return_init:
#         return train_acc_list, test_acc, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm, true_train_targets
#     else:
#         return train_acc_list, test_acc, model


# # def train_network_multiclass_spect_norm(train_loader, test_loader, depth, width, k_classes, optimizer, epochs, init_lr, decay, tqdm_flag=False, return_init=True, return_margins=False):
# #     """ Multiclass training with spectral normalization after each step"""
# #     # train_loader = full_batch_train_loader
# #     # init_lr = lr
# #     # decay = lr_decay
# #     # tqdm_flag = False
# #     # return_init = True
# #     # return_margins = True
# #     criterion = nn.MSELoss()
# #     if not tqdm_flag:
# #         tqdm = lambda x: x
# #     else:
# #         from tqdm import tqdm
# #     tqdm_  = lambda x: x
# #     model = SimpleNetMultiClass(depth, width, k_classes).cuda()
# #     # optim = Nero(model.parameters(), lr=init_lr)
# #     # optim = SGD(model.parameters(), lr=init_lr)      
# #     optim = optimizer(model.parameters(), lr=init_lr)
# #     lr_lambda = lambda x: decay**x
# #     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

# #     if return_init:
# #         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

# #     model.train()

# #     train_acc_list = []

# #     for epoch in tqdm(range(epochs)):

# #         correct = 0 
# #         total = 0

# #         for data, target in tqdm_(train_loader):
# #             data, target = (data.cuda(), target.cuda())
# #             # normalize_data trains even/odd binary 
# #             data, target = normalize_data(data, target)

# #             y_pred = model(data).squeeze()
# #             y_pred_train = y_pred.detach().cpu()

# #             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
# #             target = (target + 1).true_divide(2)
# #             target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
# #             target = target.cuda()

# #             # loss = (y_pred - target).norm()
# #             loss = criterion(y_pred.float(), target.float())
# #             _, pred_indices = y_pred.max(dim=1)
# #             _, target_indices = target.max(dim=1)
# #             correct += (pred_indices == target_indices).sum().item()
# #             # correct += (target.float() == y_pred.float().sign()).sum().item()
# #             total += target.shape[0]

# #             if epoch == epochs - 1:
# #                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
# #                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
# #                 other_class_outputs = other_class_values
# #                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])

# #             model.zero_grad()
# #             loss.backward()
# #             optim.step()
# #             # normalize spectral norms here
# #             with torch.no_grad():
# #                 # for group in model.param_groups:
# #                 for name, p in model.named_parameters():
# #                     s = np.linalg.svd(p.cpu(), full_matrices=False, compute_uv=False)
# #                     p.data = torch.true_divide(p, s[0])
# #                     split_name = name.split('.')
# #                     if ('initial' in name) or ('final' in name):
# #                         setattr(getattr(model, split_name[0]).weight, 'data', p.data)
# #                     else:
# #                         setattr(getattr(model, split_name[0])[int(split_name[1])].weight, 'data', p.data)
# #                     # setattr(getattr(model, split_name[0]), split_name[1], p.data)
                    
# #             model.cuda()
# #         lr_scheduler.step()
# #         train_acc_list.append(correct/total)

# #     model.eval()
# #     correct = 0
# #     total = 0 

# #     for data, target in tqdm_(test_loader):
# #         data, target = (data.cuda(), target.cuda())
# #         data, target = normalize_data(data, target)
        
# #         y_pred = model(data).squeeze()
# #         # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
# #         target = (target + 1).true_divide(2)
# #         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
# #         target = target.cuda()

# #         _, pred_indices = y_pred.max(dim=1)
# #         _, target_indices = target.max(dim=1)
# #         correct += (pred_indices == target_indices).sum().item()

# #         # correct += (target.float() == y_pred.float().sign()).sum().item()
# #         total += target.shape[0]

# #     if return_margins:
# #         margins = y_pred_train
# #     else:
# #         margins = None

# #     test_acc = correct/total

# #     # Compute X_norm:
# #     X, Y, W = collect_parameters_multiclass(model, train_loader)
# #     n = X.shape[0]

# #     reshaped_X = X
# #     X_norm = bartlett_X_norm(reshaped_X)

# #     if return_init:
# #         return train_acc_list, test_acc, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm
# #     else:
# #         return train_acc_list, test_acc, model


# def train_network_multiclass_combined(train_loader, test_loader, depth, width, k_classes, optimizer, epochs, init_lr, decay, to_spect_norm=False, tqdm_flag=False, return_init=True, return_margins=False):
#     """ multiclass version of train_network.  spectral normalization option included"""
#     # train_loader = full_batch_train_loader
#     # init_lr = lr
#     # decay = lr_decay
#     # tqdm_flag = False
#     # return_init = True
#     # return_margins = True
#     criterion = nn.MSELoss()
#     if not tqdm_flag:
#         tqdm = lambda x: x
#     else:
#         from tqdm import tqdm
#     tqdm_  = lambda x: x
#     model = SimpleNetMultiClass(depth, width, k_classes).cuda()
#     # optim = Nero(model.parameters(), lr=init_lr)
#     # optim = SGD(model.parameters(), lr=init_lr)      
#     optim = optimizer(model.parameters(), lr=init_lr)
#     lr_lambda = lambda x: decay**x
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

#     if return_init:
#         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

#     model.train()

#     train_acc_list = []

#     for epoch in tqdm(range(epochs)):

#         correct = 0 
#         total = 0

#         for data, target in tqdm_(train_loader):
#             data, target = (data.cuda(), target.cuda())
#             if k_classes == 10:
#                 data, target = normalize_data_10_class(data, target)
#             else:
#                 # normalize_data trains even/odd binary 
#                 data, target = normalize_data(data, target)
            

#             y_pred = model(data).squeeze()
#             y_pred_train = y_pred.detach().cpu()

#             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#             if k_classes == 2:
#                 target = (target + 1).true_divide(2)

#             target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#             target = target.cuda()

#             # loss = (y_pred - target).norm()
#             loss = criterion(y_pred.float(), target.float())
#             _, pred_indices = y_pred.max(dim=1)
#             _, target_indices = target.max(dim=1)
#             correct += (pred_indices == target_indices).sum().item()
#             # correct += (target.float() == y_pred.float().sign()).sum().item()
#             total += target.shape[0]

#             if epoch == epochs - 1:
#                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
#                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#                 other_class_outputs = other_class_values
#                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#                 true_train_targets = target

#             model.zero_grad()
#             loss.backward()
#             optim.step()

#             if to_spect_norm:
#                 with torch.no_grad():
#                     # for group in model.param_groups:
#                     for name, p in model.named_parameters():
#                         s = np.linalg.svd(p.cpu(), full_matrices=False, compute_uv=False)
#                         p.data = torch.true_divide(p, s[0])
#                         split_name = name.split('.')
#                         if ('initial' in name) or ('final' in name):
#                             setattr(getattr(model, split_name[0]).weight, 'data', p.data)
#                         else:
#                             setattr(getattr(model, split_name[0])[int(split_name[1])].weight, 'data', p.data)
#                         # setattr(getattr(model, split_name[0]), split_name[1], p.data)

#         lr_scheduler.step()
#         train_acc_list.append(correct/total)
#         # stop if perfect train acc
#         if correct/total == 1.0:
#             other_class_indices = (target_indices == 0).type(torch.LongTensor)
#             other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#             other_class_outputs = other_class_values
#             correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#             true_train_targets = target
#             break

#     model.eval()
#     correct = 0
#     total = 0 

#     for data, target in tqdm_(test_loader):
#         data, target = (data.cuda(), target.cuda())
#         if k_classes == 10:
#             data, target = normalize_data_10_class(data, target)
#         else:
#             # normalize_data trains even/odd binary 
#             data, target = normalize_data(data, target)
        
#         y_pred = model(data).squeeze()
#         # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#         if k_classes == 2:
#             target = (target + 1).true_divide(2)

#         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#         target = target.cuda()

#         _, pred_indices = y_pred.max(dim=1)
#         _, target_indices = target.max(dim=1)
#         correct += (pred_indices == target_indices).sum().item()

#         # correct += (target.float() == y_pred.float().sign()).sum().item()
#         total += target.shape[0]

#     if return_margins:
#         margins = y_pred_train
#     else:
#         margins = None

#     test_acc = correct/total
    
#     # Compute X_norm:
#     X, Y, W = collect_parameters_multiclass(model, train_loader)
#     n = X.shape[0]

#     reshaped_X = X
#     X_norm = bartlett_X_norm(reshaped_X)

#     if return_init:
#         return train_acc_list, test_acc, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm, true_train_targets
#     else:
#         return train_acc_list, test_acc, model


# def train_network_multiclass_scale_label(train_loader, test_loader, depth, width, k_classes, optimizer, epochs, init_lr, decay, to_spect_norm=False, label_scale=1, return_init=True, return_margins=False):
#     criterion = nn.MSELoss()
#     from tqdm import tqdm
#     tqdm_  = lambda x: x

#     model = SimpleNetMultiClass(depth, width, k_classes).cuda()

#     optim = optimizer(model.parameters(), lr=init_lr)
#     lr_lambda = lambda x: decay**x
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

#     if return_init:
#         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

#     model.train()

#     train_acc_list = []

#     for epoch in tqdm(range(epochs)):

#         correct = 0 
#         total = 0

#         for data, target in tqdm_(train_loader):
#             data, target = (data.cuda(), target.cuda())
#             if k_classes == 10:
#                 data, target = normalize_data_10_class(data, target)
#             else:
#                 # normalize_data trains even/odd binary 
#                 data, target = normalize_data(data, target)
        

#             y_pred = model(data).squeeze()
#             y_pred_train = y_pred.detach().cpu()

#             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#             if k_classes == 2:
#                 target = (target + 1).true_divide(2)

#             target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#             target *= label_scale
#             target = target.cuda()

#             # loss = (y_pred - target).norm()
#             loss = criterion(y_pred.float(), target.float())
#             _, pred_indices = y_pred.max(dim=1)
#             _, target_indices = target.max(dim=1)
#             correct += (pred_indices == target_indices).sum().item()
#             # correct += (target.float() == y_pred.float().sign()).sum().item()
#             total += target.shape[0]

#             if epoch == epochs - 1:
#                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
#                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#                 other_class_outputs = other_class_values
#                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#                 true_train_targets = target

#             model.zero_grad()
#             loss.backward()
#             optim.step()

#             if to_spect_norm:
#                 with torch.no_grad():
#                     # for group in model.param_groups:
#                     for name, p in model.named_parameters():
#                         s = np.linalg.svd(p.cpu(), full_matrices=False, compute_uv=False)
#                         p.data = torch.true_divide(p, s[0])
#                         split_name = name.split('.')
#                         if ('initial' in name) or ('final' in name):
#                             setattr(getattr(model, split_name[0]).weight, 'data', p.data)
#                         else:
#                             setattr(getattr(model, split_name[0])[int(split_name[1])].weight, 'data', p.data)
#                         # setattr(getattr(model, split_name[0]), split_name[1], p.data)

#         lr_scheduler.step()
#         train_acc_list.append(correct/total)
#         # stop if perfect train acc
#         if correct/total == 1.0:
#             other_class_indices = (target_indices == 0).type(torch.LongTensor)
#             other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#             other_class_outputs = other_class_values
#             correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#             true_train_targets = target
#             break

#     model.eval()
#     correct = 0
#     total = 0 

#     for data, target in tqdm_(test_loader):
#         data, target = (data.cuda(), target.cuda())
#         if k_classes == 10:
#             data, target = normalize_data_10_class(data, target)
#         else:
#             # normalize_data trains even/odd binary 
#             data, target = normalize_data(data, target)
        
#         y_pred = model(data).squeeze()
#         # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#         if k_classes == 2:
#             target = (target + 1).true_divide(2)

#         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#         target = target.cuda()

#         _, pred_indices = y_pred.max(dim=1)
#         _, target_indices = target.max(dim=1)
#         correct += (pred_indices == target_indices).sum().item()

#         # correct += (target.float() == y_pred.float().sign()).sum().item()
#         total += target.shape[0]

#     if return_margins:
#         margins = y_pred_train
#     else:
#         margins = None

#     test_acc = correct/total
    
#     # Compute X_norm:
#     X, Y, W = collect_parameters_multiclass(model, train_loader)
#     n = X.shape[0]

#     reshaped_X = X
#     X_norm = bartlett_X_norm(reshaped_X)

#     if return_init:
#         return train_acc_list, test_acc, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm, true_train_targets
#     else:
#         return train_acc_list, test_acc, model


# def train_network_multiclass_scale_label_input_net(model, criterion, train_loader, test_loader, depth, width, k_classes, optimizer, epochs, init_lr, decay, to_spect_norm=False, label_scale=1, return_init=True, return_margins=False):
#     """Same as train_network_multiclass_scale_label_input except can input a 
#     network.  For use with extreme memorization work.
#     This variation will compute test accuracy after every epoch.
#     """
#     # criterion = nn.MSELoss()
#     from tqdm import tqdm
#     tqdm_  = lambda x: x

#     # model = SimpleNetMultiClass(depth, width, k_classes).cuda()
#     model = model.cuda()

#     optim = optimizer(model.parameters(), lr=init_lr)
#     lr_lambda = lambda x: decay**x
#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

#     if return_init:
#         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

#     model.train()

#     train_acc_list = []
#     test_acc_list = []

#     for epoch in tqdm(range(epochs)):

#         correct = 0 
#         total = 0

#         for data, target in tqdm_(train_loader):
#             data, target = (data.cuda(), target.cuda())
#             if k_classes == 10:
#                 data, target = normalize_data_10_class(data, target)
#             else:
#                 # normalize_data trains even/odd binary 
#                 data, target = normalize_data(data, target)
        

#             y_pred = model(data).squeeze()
#             y_pred_train = y_pred.detach().cpu()

#             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#             if k_classes == 2:
#                 target = (target + 1).true_divide(2)
#             orig_target = target
#             target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#             # target *= label_scale # problems with casting float as long
#             target = target * label_scale
#             target = target.cuda()

#             # loss = (y_pred - target).norm()
#             if type(nn.CrossEntropyLoss()) == type(criterion):
#                 orig_target = orig_target.cuda()
#                 loss = criterion(y_pred, orig_target)
#             else:
#                 loss = criterion(y_pred.float(), target.float())
#             _, pred_indices = y_pred.max(dim=1)
#             _, target_indices = target.max(dim=1)
#             correct += (pred_indices == target_indices).sum().item()
#             # correct += (target.float() == y_pred.float().sign()).sum().item()
#             total += target.shape[0]

#             if epoch == epochs - 1:
#                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
#                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#                 other_class_outputs = other_class_values
#                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#                 true_train_targets = target

#             model.zero_grad()
#             loss.backward()
#             optim.step()

#             if to_spect_norm:
#                 with torch.no_grad():
#                     # for group in model.param_groups:
#                     for name, p in model.named_parameters():
#                         s = np.linalg.svd(p.cpu(), full_matrices=False, compute_uv=False)
#                         p.data = torch.true_divide(p, s[0])
#                         split_name = name.split('.')
#                         if ('initial' in name) or ('final' in name):
#                             setattr(getattr(model, split_name[0]).weight, 'data', p.data)
#                         else:
#                             setattr(getattr(model, split_name[0])[int(split_name[1])].weight, 'data', p.data)
#                         # setattr(getattr(model, split_name[0]), split_name[1], p.data)
#         lr_scheduler.step()
#         train_acc_list.append(correct/total)
#         train_correct = correct
#         train_total = total
#         # Here we compute the test accuracy after every epoch
#         with torch.no_grad():
#             for data, target in tqdm_(test_loader):
#                     data, target = (data.cuda(), target.cuda())
#                     if k_classes == 10:
#                         data, target = normalize_data_10_class(data, target)
#                     else:
#                         # normalize_data trains even/odd binary 
#                         data, target = normalize_data(data, target)
                    
#                     y_pred = model(data).squeeze()
#                     # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#                     if k_classes == 2:
#                         target = (target + 1).true_divide(2)

#                     target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#                     target = target.cuda()

#                     _, pred_indices = y_pred.max(dim=1)
#                     _, target_indices = target.max(dim=1)
#                     correct += (pred_indices == target_indices).sum().item()

#                     # correct += (target.float() == y_pred.float().sign()).sum().item()
#                     total += target.shape[0]

#             if return_margins:
#                 margins = y_pred_train
#             else:
#                 margins = None

#         test_acc = correct/total
#         test_acc_list.append(correct/total)
        
#         # Turn off train-early stopping block for extreme memorization work
#         # # stop if perfect train acc
#         # if train_correct/train_total == 1.0:
#         #     other_class_indices = (target_indices == 0).type(torch.LongTensor)
#         #     other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#         #     other_class_outputs = other_class_values
#         #     correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#         #     true_train_targets = target
#         #     break

#     model.eval()
#     correct = 0
#     total = 0 
# #   WE ARE ALSO TURNING OFF FINAL TEST ACCURACY CALCULATION
#     # for data, target in tqdm_(test_loader):
#     #     data, target = (data.cuda(), target.cuda())
#     #     if k_classes == 10:
#     #         data, target = normalize_data_10_class(data, target)
#     #     else:
#     #         # normalize_data trains even/odd binary 
#     #         data, target = normalize_data(data, target)
        
#     #     y_pred = model(data).squeeze()
#     #     # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#     #     if k_classes == 2:
#     #         target = (target + 1).true_divide(2)

#     #     target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#     #     target = target.cuda()

#     #     _, pred_indices = y_pred.max(dim=1)
#     #     _, target_indices = target.max(dim=1)
#     #     correct += (pred_indices == target_indices).sum().item()

#     #     # correct += (target.float() == y_pred.float().sign()).sum().item()
#     #     total += target.shape[0]

#     # if return_margins:
#     #     margins = y_pred_train
#     # else:
#     #     margins = None

#     # test_acc = correct/total
    
#     # Compute X_norm:
#     X, Y, W = collect_parameters_multiclass(model, train_loader)
#     n = X.shape[0]

#     reshaped_X = X
#     X_norm = bartlett_X_norm(reshaped_X)

#     if return_init:
#         return train_acc_list, test_acc_list, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm, true_train_targets
#     else:
#         return train_acc_list, test_acc_list, model


# def generalized_multiclass_train(model, criterion, train_loader, test_loader,
#                                  k_classes, optimizer, optimizer_kwargs,
#                                  epochs, lr_decay, to_spect_norm=False,
#                                 label_scale=1, return_init=True,
#                                 return_margins=False, early_stop=False,
#                                 tqdm_flag=False):
#     """Generalized version of train_network_multiclass_scale_label_input.

#     model: SimpleNet* variant with the appropriate number of k_classes
#     criterion: 
#     network.  For use with extreme memorization work.
#     This variation will compute test accuracy after every epoch.
#     """

#     #TODO: implement kwargs for optimizer, and possibly model. Can input the model type and then a dictionary of kwargs to match up with it
#     from tqdm import tqdm
#     tqdm_  = lambda x: x
#     tqdm__ = lambda x: x
#     if tqdm_flag:
#         tqdm_ = tqdm
        
#     model = model.cuda()
#     optimizer_kwargs['params'] = model.parameters()
#     optim = optimizer(**optimizer_kwargs)
#     # optim = optimizer(model.parameters(), lr=init_lr)
#     # Changing the scheduler to exponential

#     # lr_lambda = lambda x: lr_decay**x
#     # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
#     lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)


#     if return_init:
#         init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

#     model.train()

#     train_acc_list = []
#     test_acc_list = []

#     for epoch in tqdm_(range(epochs)):

#         correct = 0 
#         total = 0

#         for data, target in tqdm__(train_loader):
#             data, target = (data.cuda(), target.cuda())
#             if k_classes == 10:
#                 data, target = normalize_data_10_class(data, target)
#             else:
#                 # normalize_data trains even/odd binary 
#                 data, target = normalize_data(data, target)
        

#             y_pred = model(data).squeeze()
#             y_pred_train = y_pred.detach().cpu()

#             # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#             if k_classes == 2:
#                 target = (target + 1).true_divide(2)
#             orig_target = target
#             target = torch.nn.functional.one_hot(target.type(torch.LongTensor), num_classes=k_classes)
#             # target *= label_scale # problems with casting float as long
#             target = target * label_scale
#             target = target.cuda()

#             # loss = (y_pred - target).norm()
#             if type(nn.CrossEntropyLoss()) == type(criterion):
#                 orig_target = orig_target.cuda()
#                 loss = criterion(y_pred, orig_target)
#             else:
#                 loss = criterion(y_pred.float(), target.float())
#             _, pred_indices = y_pred.max(dim=1)
#             _, target_indices = target.max(dim=1)
#             correct += (pred_indices == target_indices).sum().item()
#             # correct += (target.float() == y_pred.float().sign()).sum().item()
#             total += target.shape[0]

#             if epoch == epochs - 1:
#                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
#                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#                 other_class_outputs = other_class_values
#                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#                 true_train_targets = target

#             model.zero_grad()
#             loss.backward()
#             optim.step()

#             if to_spect_norm:
#                 with torch.no_grad():
#                     # for group in model.param_groups:
#                     for name, p in model.named_parameters():
#                         s = np.linalg.svd(p.cpu(), full_matrices=False, compute_uv=False)
#                         p.data = torch.true_divide(p, s[0])
#                         split_name = name.split('.')
#                         if ('initial' in name) or ('final' in name):
#                             setattr(getattr(model, split_name[0]).weight, 'data', p.data)
#                         else:
#                             setattr(getattr(model, split_name[0])[int(split_name[1])].weight, 'data', p.data)
#                         # setattr(getattr(model, split_name[0]), split_name[1], p.data)
#         lr_scheduler.step()
#         train_acc_list.append(correct/total)
#         train_correct = correct
#         train_total = total

#         model.eval()
#         correct = 0 
#         total = 0
#         # Here we compute the test accuracy after every epoch
#         with torch.no_grad():
#             for data, target in tqdm_(test_loader):
#                     data, target = (data.cuda(), target.cuda())
#                     if k_classes == 10:
#                         data, target = normalize_data_10_class(data, target)
#                     else:
#                         # normalize_data trains even/odd binary 
#                         data, target = normalize_data(data, target)
                    
#                     y_pred = model(data).squeeze()
#                     # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
#                     if k_classes == 2:
#                         target = (target + 1).true_divide(2)

#                     target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
#                     target = target.cuda()

#                     _, pred_indices = y_pred.max(dim=1)
#                     _, target_indices = target.max(dim=1)
#                     correct += (pred_indices == target_indices).sum().item()

#                     # correct += (target.float() == y_pred.float().sign()).sum().item()
#                     total += target.shape[0]

#             if return_margins:
#                 margins = y_pred_train
#             else:
#                 margins = None

#         test_acc = correct/total
#         test_acc_list.append(correct/total)
        
#         # Turn off train-early stopping block for extreme memorization work
#         # # stop if perfect train acc
#         if early_stop:
#             if train_correct/train_total == 1.0:
#                 other_class_indices = (target_indices == 0).type(torch.LongTensor)
#                 other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
#                 other_class_outputs = other_class_values
#                 correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
#                 true_train_targets = target
#                 break

#     # model.eval()
#     # correct = 0
#     # total = 0 
#     X, Y, W = collect_parameters_multiclass(model, train_loader)
#     n = X.shape[0]

#     reshaped_X = X
#     X_norm = bartlett_X_norm(reshaped_X)

#     if return_init:
#         return train_acc_list, test_acc_list, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm, true_train_targets
#     else:
#         return train_acc_list, test_acc_list, model


def generalized_multiclass_train_fullbatch(model, criterion, train_loader, test_loader,
                                 k_classes, optimizer, optimizer_kwargs,
                                 epochs, lr_decay, to_spect_norm=False,
                                label_scale=1, return_init=True,
                                return_margins=False, early_stop=False,
                                tqdm_flag=False):
    """Generalized version of train_network_multiclass_scale_label_input.

    model: SimpleNet* variant with the appropriate number of k_classes
    criterion: 
    network.  For use with extreme memorization work.
    This variation will compute test accuracy after every epoch.
    """

    #TODO: implement kwargs for optimizer, and possibly model. Can input the model type and then a dictionary of kwargs to match up with it
    from tqdm import tqdm
    tqdm_  = lambda x: x
    tqdm__ = lambda x: x
    if tqdm_flag:
        tqdm_ = tqdm
        
    model = model.cuda()
    optimizer_kwargs['params'] = model.parameters()
    optim = optimizer(**optimizer_kwargs)
    # optim = optimizer(model.parameters(), lr=init_lr)
    # Changing the scheduler to exponential

    # lr_lambda = lambda x: lr_decay**x
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)


    if return_init:
        init_weights = [x.detach().cpu().numpy() for x in model.parameters()]

    model.train()

    train_acc_list = []
    test_acc_list = []

    # loading once the data:
    for data, target in tqdm__(train_loader):
        data, target = (data.cuda(), target.cuda())
    if k_classes == 10:
        data, target = normalize_data_10_class(data, target)
    else:
        # normalize_data trains even/odd binary 
        data, target = normalize_data(data, target)
    
    if k_classes == 2:
        target = (target+1).true_divide(2)
    orig_target = target
    target = torch.nn.functional.one_hot(target.type(torch.LongTensor), num_classes=k_classes)
    target = target * label_scale
    target = target.cuda()

    if type(nn.CrossEntropyLoss()) == type(criterion):
        orig_target = orig_target.cuda()
        # loss = criterion(y_pred, orig_target)
    else:
        pass
        # loss = criterion(y_pred.float(), target.float())
    _, target_indices = target.max(dim=1)

    # Load test data:
    for data_test, target_test in tqdm__(test_loader):
        data_test, target_test = (data_test.cuda(), target_test.cuda())
    if k_classes == 10:
        data_test, target_test = normalize_data_10_class(data_test, target_test)
    else:
        data_test, target_test = normalize_data(data_test, target_test)
    
    if k_classes == 2:
        target_test = (target_test+1).true_divide(2)
    orig_target_test = target_test
    target_test = torch.nn.functional.one_hot(target_test.type(torch.LongTensor), num_classes=k_classes)
    if type(nn.CrossEntropyLoss()) == type(criterion):
        orig_target_test = orig_target_test.cuda()
        # loss = criterion(y_pred, orig_target)
    else:
        pass
        # loss = criterion(y_pred.float(), target.float())
    _, target_test_indices = target_test.max(dim=1)
    target_test_indices = target_test_indices.cuda()
    

    for epoch in tqdm_(range(epochs)):

        correct = 0 
        total = 0

        # for data, target in tqdm__(train_loader):
        #     data, target = (data.cuda(), target.cuda())
        #     if k_classes == 10:
        #         data, target = normalize_data_10_class(data, target)
        #     else:
        #         # normalize_data trains even/odd binary 
        #         data, target = normalize_data(data, target)
        

        y_pred = model(data).squeeze()
        y_pred_train = y_pred.detach().cpu()

            # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
            # if k_classes == 2:
            #     target = (target + 1).true_divide(2)
            # orig_target = target
            # target = torch.nn.functional.one_hot(target.type(torch.LongTensor), num_classes=k_classes)
            # target *= label_scale # problems with casting float as long
            # target = target * label_scale
            # target = target.cuda()

            # loss = (y_pred - target).norm()
        if type(nn.CrossEntropyLoss()) == type(criterion):
            # orig_target = orig_target.cuda()
            loss = criterion(y_pred, orig_target)
        else:
            # print(f'y_pred.is_cuda: {y_pred.is_cuda}, target.is_cuda: {target.is_cuda}')
            loss = criterion(y_pred.float(), target.float())
        _, pred_indices = y_pred.max(dim=1)
        # _, target_indices = target.max(dim=1)
        correct += (pred_indices == target_indices).sum().item()
            # correct += (target.float() == y_pred.float().sign()).sum().item()
        total += target.shape[0]

        if epoch == epochs - 1:
            other_class_indices = (target_indices == 0).type(torch.LongTensor)
            other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
            other_class_outputs = other_class_values
            correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
            true_train_targets = target

        model.zero_grad()
        loss.backward()
        optim.step()

        if to_spect_norm:
            with torch.no_grad():
                # for group in model.param_groups:
                for name, p in model.named_parameters():
                    s = np.linalg.svd(p.cpu(), full_matrices=False, compute_uv=False)
                    p.data = torch.true_divide(p, s[0])
                    split_name = name.split('.')
                    if ('initial' in name) or ('final' in name):
                        setattr(getattr(model, split_name[0]).weight, 'data', p.data)
                    else:
                        setattr(getattr(model, split_name[0])[int(split_name[1])].weight, 'data', p.data)
                    # setattr(getattr(model, split_name[0]), split_name[1], p.data)
        lr_scheduler.step()
        train_acc_list.append(correct/total)
        train_correct = correct
        train_total = total

        model.eval()
        correct = 0 
        total = 0
        # Here we compute the test accuracy after every epoch
        with torch.no_grad():
            # for data, target in tqdm_(test_loader):
            #         data, target = (data.cuda(), target.cuda())
            #         if k_classes == 10:
            #             data, target = normalize_data_10_class(data, target)
            #         else:
            #             # normalize_data trains even/odd binary 
            #             data, target = normalize_data(data, target)
                    
            #         y_pred = model(data).squeeze()
            #         # Convert the -1,+1 encoding to 0,1 classes and then to one hot [1,0] [0,1]
            #         if k_classes == 2:
            #             target = (target + 1).true_divide(2)

            #         target = torch.nn.functional.one_hot(target.type(torch.LongTensor))
            #         target = target.cuda()

            #         _, pred_indices = y_pred.max(dim=1)
            #         _, target_indices = target.max(dim=1)
            #         correct += (pred_indices == target_indices).sum().item()

            #         # correct += (target.float() == y_pred.float().sign()).sum().item()
            #         total += target.shape[0]
            y_pred_test = model(data_test).squeeze()
            _, pred_test_indices = y_pred_test.max(dim=1)
            # print(f"pred_test_indices: {pred_test_indices.is_cuda}, target_test_indices: {target_test_indices.is_cuda}")
            correct += (pred_test_indices == target_test_indices).sum().item()
            total += target_test.shape[0]


            if return_margins:
                margins = y_pred_train
            else:
                margins = None

        test_acc = correct/total
        test_acc_list.append(correct/total)
        
        # Turn off train-early stopping block for extreme memorization work
        # # stop if perfect train acc
        if early_stop:
            if train_correct/train_total == 1.0:
                other_class_indices = (target_indices == 0).type(torch.LongTensor)
                other_class_values  = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(other_class_indices)])
                other_class_outputs = other_class_values
                correct_class_outputs = torch.Tensor([y_pred[i, index].detach().cpu() for i, index in enumerate(target_indices)])
                true_train_targets = target
                break

    # model.eval()
    # correct = 0
    # total = 0 
    X, Y, W = collect_parameters_multiclass(model, train_loader)
    n = X.shape[0]

    reshaped_X = X
    X_norm = bartlett_X_norm(reshaped_X)

    if return_init:
        return train_acc_list, test_acc_list, model, init_weights, margins, correct_class_outputs, other_class_outputs, X_norm, true_train_targets
    else:
        return train_acc_list, test_acc_list, model