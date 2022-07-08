import torch
from torch.optim.optimizer import Optimizer
# Code taken from and modified from https://github.com/jxbz/nero
# from the paper Learning by Turning: Neural Architecture Aware Optimisation
# https://arxiv.org/abs/2102.07227

def neuron_norm(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.norm(dim=1).view(*view_shape)
    else:
        return x.abs()

def neuron_mean(x):
    if x.dim() > 1:
        view_shape = [x.shape[0]] + [1]*(x.dim()-1)
        x = x.view(x.shape[0],-1)
        return x.mean(dim=1).view(*view_shape)
    else:
        raise Exception("neuron_mean not defined on 1D tensors.")

class Nero(Optimizer):

    def __init__(self, params, lr=0.01, beta=0.999, constraints=True):
        self.beta = beta
        self.constraints = constraints
        defaults = dict(lr=lr)
        super(Nero, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)
                state = self.state[p]
                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
                state['scale'] = neuron_norm(p).mean()
                if state['scale'] == 0.0:
                    state['scale'] = 0.01

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                state['step'] += 1
                bias_correction = 1 - self.beta ** state['step']
                state['exp_avg_sq'] = self.beta * state['exp_avg_sq'] + (1-self.beta) * neuron_norm(p.grad)**2

                grad_normed = p.grad / (state['exp_avg_sq']/bias_correction).sqrt()
                grad_normed[torch.isnan(grad_normed)] = 0
                
                p.data -= group['lr'] * state['scale'] * grad_normed

                if self.constraints and p.dim() > 1:
                    p.data -= neuron_mean(p)
                    p.data /= neuron_norm(p)

        return loss


# class ScaledNero(Optimizer):
#     """A version of Nero that allows scaling of the neuron norm to allow
#     different scales of initialization but also allow for a scaled gradient.  
    
#     Setting constraints to false should scale the gradient but not project
#     the row norms to the sigma_scale * unit ball."""

#     def __init__(self, params, sigma_scale=1, lr=0.01, beta=0.999, constraints=True):
#         self.beta = beta
#         self.constraints = constraints
#         self.sigma_scale = sigma_scale
#         defaults = dict(lr=lr)
#         super(ScaledNero, self).__init__(params, defaults)

#         for group in self.param_groups:
#             for p in group['params']:
#                 if self.constraints and p.dim() > 1:
#                     p.data -= neuron_mean(p)
#                     p.data /= neuron_norm(p)
#                     p.data *= sigma_scale
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg_sq'] = torch.zeros_like(neuron_norm(p))
#                 state['scale'] = neuron_norm(p).mean()
#                 if state['scale'] == 0.0:
#                     state['scale'] = 0.01

#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 state = self.state[p]

#                 state['step'] += 1
#                 bias_correction = 1 - self.beta ** state['step']
#                 state['exp_avg_sq'] = self.beta * state['exp_avg_sq'] + (1-self.beta) * neuron_norm(p.grad)**2

#                 grad_normed = p.grad / (state['exp_avg_sq']/bias_correction).sqrt()
#                 grad_normed[torch.isnan(grad_normed)] = 0
                
#                 p.data -= group['lr'] * state['scale'] * grad_normed

#                 if self.constraints and p.dim() > 1:
#                     p.data -= neuron_mean(p)
#                     p.data /= neuron_norm(p)
#                     p.data *= self.sigma_scale

#         return loss
