import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class optimizer:
    """
    Parent class for all optimizers
    """
    def __init__(self, parameters, device):
        self.parameters = list(parameters)
        self.device = device

    @abstractmethod
    def step(self):
        """
        Updates parameters
        """
        pass

    def zero_grad(self):
        """
        Sets gradients to zeros 
        """
        for p in self.parameters:
            p.grad = None


class SGDOptimizer(optimizer):
    """
    Vanilla Minibatch Stochastic Gradient Descent 
    """
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.learning_rate = args["learning_rate"]  # learning rate

    def step(self):
        for p in self.parameters:
            # p.grad - minibatch gradient 
            p.data -= p.grad * self.learning_rate  # update parameters 


class MomentumSGDOptimizer(optimizer):
    """
    Stochastic Gradient decsent with momentum 
    """
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.learning_rate = args["learning_rate"]  # learning rate 
        self.rho = args["rho"]  # momentum 
        self.m = None

    def step(self):
        if self.m is None:  # initialize velocity 
            self.m = [torch.zeros(p.size()).to(self.device) for p in self.parameters]  # initialize velocity 

        for i, p in enumerate(self.parameters):
            self.m[i] = self.rho * self.m[i] + p.grad  # update velocity 
            p.grad = self.learning_rate * self.m[i]  
            p.data -= p.grad  # update parameters 


class RMSPropOptimizer(optimizer):
    """
    RMSProp algorithm: 
    Divide the learning rate by average gradient norm in diagonal 
    """ 
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.tau = args["tau"]  # decaying parameter
        self.learning_rate = args["learning_rate"]  # global learning rate
        self.delta = args["delta"]  # damping coefficient
        self.r = None

    def step(self):
        if self.r is None:
            self.r = [torch.zeros(p.size()).to(self.device) for p in self.parameters]  # initialize past gradient information 

        for i, p in enumerate(self.parameters):
            grad = p.grad  # minibatch gradient 
            self.r[i] = self.r[i] * self.tau + (1 - self.tau) * grad * grad  # recent gradients have greater importance 
            p.data -= self.learning_rate / (self.delta + torch.sqrt(self.r[i])) * grad  # update parameters 


class AMSgradOptimizer(optimizer):
    """
    AMSgrad algorithm: 
    Uses maximum to update parameters, so it converges for some functions when ADAM doesn't
    """
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.beta1 = args["beta1"]  # 1st order decaying parameter
        self.beta2 = args["beta2"]  # 2d order decaying parameter
        self.learning_rate = args["learning_rate"]  # global learning rate
        self.delta = args["delta"]  # damping coefficient
        
        self.iteration = None
        self.m1 = None
        self.m2 = None
        self.m2_max = None 

    def step(self):
        # initialization 
        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.m2_max is None:
            self.m2_max = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(self.parameters):
            grad = p.grad  # minibatch gradient 
            self.m1[i] = self.m1[i] * self.beta1 + (1 - self.beta1) * grad  # 1st order estimation 
            self.m2[i] = self.m2[i] * self.beta2 + (1 - self.beta2) * grad * grad  # 2d order estimation 
            m1_hat = self.m1[i] / (1 - self.beta1 ** self.iteration)  # bias correction 
            m2_hat = self.m2[i] / (1 - self.beta2 ** self.iteration)  # bias correction 
            self.m2_max[i] = torch.maximum(m2_hat, self.m2_max[i])  # update maximum 
            p.data -= self.learning_rate * m1_hat / (self.delta + torch.sqrt(self.m2_max[i]))  # update parameters 

        self.iteration = self.iteration + 1


class AdagradOptimizer(optimizer):
    """
    Adagrad algorithm: 
    By estimating local geometry adapts step-size for every coordinate 
    """
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.learning_rate = args["learning_rate"]  # global learning rate
        self.delta = args["delta"]  # damping coefficient
        self.r = None

    def step(self):
        if self.r is None:
            self.r = [torch.zeros(p.size()).to(self.device) for p in self.parameters]  # initialize gradient norm

        for i, p in enumerate(self.parameters):
            grad = p.grad  # minibatch grdaient 
            self.r[i] = self.r[i] + grad * grad  # calculate gradient norm 
            p.data -= self.learning_rate / (self.delta + torch.sqrt(self.r[i])) * grad  # update parameters 


class ADAMOptimizer(optimizer):
    """
    ADAM algorithm: 
    Combination of RMSprop and 2d order estimation leads to variance reduction
    """
    def __init__(self, parameters, device, args):
        super().__init__(parameters, device)
        self.beta1 = args["beta1"]  # 1st order decaying parameter
        self.beta2 = args["beta2"]  # 2d order decaying parameter
        self.learning_rate = args["learning_rate"]  # global learning rate
        self.delta = args["delta"]  # damping coefficient
        
        self.iteration = None
        self.m = None
        self.v = None

    def step(self):
        # initialization 
        if self.m is None:
            self.m = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.v is None:
            self.v = [torch.zeros(p.grad.size()).to(self.device) for p in self.parameters]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(self.parameters):
            grad = p.grad  # minibatch gradient 
            self.m[i] = self.m[i] * self.beta1 + (1 - self.beta1) * grad  # 1st order estimation
            self.v[i] = self.v[i] * self.beta2 + (1 - self.beta2) * grad * grad  # 2d order estimation
            m_hat = self.m[i] / (1 - self.beta1 ** self.iteration)  # bias correction 
            v_hat = self.v[i] / (1 - self.beta2 ** self.iteration)  # bias correction 
            p.data -= self.learning_rate * m_hat / (self.delta + torch.sqrt(v_hat))   # update parameters 

        self.iteration = self.iteration + 1


def createOptimizer(device, args, model):
    """
    Returns the optimizer by a key word 
    from the list ["sgd", "momentumsgd", "adagrad", "adam", "rmsprop", "amsgrad"].
    :params device - "cuda" or "cpu"; 
    :params args - dict of optimizers' hyperparameters; 
    :params model - PyTorch model to optimize. 
    """
    p = model.parameters()
    if args["optimizer"] == "sgd":
        return SGDOptimizer(p, device, args)
    elif args["optimizer"] == "momentumsgd":
        return MomentumSGDOptimizer(p, device, args)
    elif args["optimizer"] == "adagrad":
        return AdagradOptimizer(p, device, args)
    elif args["optimizer"] == "adam":
        return ADAMOptimizer(p, device, args)
    elif args["optimizer"] == "rmsprop":
        return RMSPropOptimizer(p, device, args)
    elif args["optimizer"] == "amsgrad":
        return AMSgradOptimizer(p, device, args)
    else:
        raise NotImplementedError(f"Unknown optimizer {args['optimizer']}")
