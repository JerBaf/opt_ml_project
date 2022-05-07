from abc import abstractmethod
import torch


class optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    @abstractmethod
    def step(self):
        """
        Perform one step of the optimizer using the gradient supplied by `loss.backward()`
        """
        pass

    def zero_grad(self):
        """
    Zero the gradient of each parameter
        """
        for p in self.parameters:
            p.grad = None


class SGDOptimizer(optimizer):
    def __init__(self, parameters, lr):
        super().__init__(parameters)
        self.learning_rate = lr

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.learning_rate


class MomentumSGDOptimizer(optimizer):
    def __init__(self, parameters, args):
        super().__init__(parameters)
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.m = None

    def step(self):
        if self.m is None:
            self.m = [torch.zeros(p.size()) for p in self.parameters]

        for i, p in enumerate(self.parameters):
            self.m[i] = self.rho * self.m[i] + p.grad
            p.grad = self.learning_rate * self.m[i]
            p.data -= p.grad


class RMSPropOptimizer(optimizer):
    def __init__(self, parameters, args):
        super().__init__(parameters)
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.r = None
        self.delta = args.delta

    def step(self):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in self.parameters]

        for i, p in enumerate(self.parameters):
            grad = p.grad
            self.r[i] = self.r[i] * self.tau + (1 - self.tau) * grad * grad
            p.data -= self.learning_rate / (self.delta + torch.sqrt(self.r[i])) * grad
            # raise NotImplementedError('You should write your code HERE')


class AMSgradOptimizer(optimizer):
    def __init__(self, parameters, args):
        super().__init__(parameters)
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.learning_rate = args.learning_rate
        self.delta = args.delta
        self.iteration = None
        self.m1 = None
        self.m2 = None
        self.m2_max = None  # Maintains max of all exp. moving avg. of sq. grad. values

    def step(self):

        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()) for p in self.parameters]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()) for p in self.parameters]
        if self.m2_max is None:
            self.m2_max = [torch.zeros(p.grad.size()) for p in self.parameters]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(self.parameters):
            grad = p.grad
            self.m1[i] = self.m1[i] * self.beta1 + (1 - self.beta1) * grad
            self.m2[i] = self.m2[i] * self.beta2 + (1 - self.beta2) * grad * grad
            m1_hat = self.m1[i] / (1 - self.beta1 ** self.iteration)
            m2_hat = self.m2[i] / (1 - self.beta2 ** self.iteration)
            self.m2_max[i] = torch.maximum(m2_hat, self.m2_max[i])
            p.data -= self.learning_rate * m1_hat / (self.delta + torch.sqrt(self.m2_max[i]))
            # raise NotImplementedError('You should write your code HERE')

        self.iteration = self.iteration + 1


def createOptimizer(args, model):
    p = model.parameters()
    if args.optimizer == "sgd":
        return SGDOptimizer(p, args)
    elif args.optimizer == "momentumsgd":
        return MomentumSGDOptimizer(p, args)
    #     elif args.optimizer == "adagrad":
    #         return AdagradOptimizer(p,args)
    elif args.optimizer == "rmsprop":
        return RMSPropOptimizer(p, args)
    elif args.optimizer == "amsgrad":
        return AMSgradOptimizer(p, args)
    else:
        raise NotImplementedError(f"Unknown optimizer {args.optimizer}")
