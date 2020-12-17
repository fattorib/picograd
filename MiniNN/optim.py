from MiniNN.Tensor import *


class Optimizer():

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


class SGD(Optimizer):

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.value = parameter.value - (parameter.grad)*self.lr


class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-08):
        # Using default Adam params
        self.parameters = parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Need one of each per layer
        self.m = [np.zeros(param.shape) for param in self.parameters]
        self.v = [np.zeros(param.shape) for param in self.parameters]
        self.eps = eps
        self.step_num = 0

    def adam_step(self, parameter_idx):
        grad = self.parameters[parameter_idx].grad
        self.m[parameter_idx] = self.beta_1 * \
            self.m[parameter_idx] + (1-self.beta_1)*grad

        self.v[parameter_idx] = self.beta_2 * \
            self.v[parameter_idx] + (1-self.beta_1)*(grad**2)

        m_corrected = self.m[parameter_idx]/(1-(self.beta_1**self.step_num))
        v_corrected = self.v[parameter_idx]/(1-(self.beta_2**self.step_num))

        return self.lr*m_corrected/(np.sqrt(v_corrected) + self.eps)

    def step(self):
        self.step_num += 1
        for parameter_idx in range(0, len(self.parameters)):
            self.parameters[parameter_idx].value = self.parameters[parameter_idx].value - \
                self.adam_step(parameter_idx)
