
import math
import torch
from torch import nn

def init_params(weights, bias=None):

    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

def get_optimizer(params, lr, momentum=0, weight_decay=0, use_adam=False,
                  adam_beta1=0.9, use_rmsprop=False, use_adadelta=False,
                  use_adagrad=False):

    if use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, betas=[adam_beta1, 0.999],
                                     weight_decay=weight_decay)
    elif use_rmsprop:
        optimizer = torch.optim.RMSprop(params, lr=lr,
                                        weight_decay=weight_decay,
                                        momentum=momentum)
    elif use_adadelta:
        optimizer = torch.optim.Adadelta(params, lr=lr,
                                         weight_decay=weight_decay)
    elif use_adagrad:
        optimizer = torch.optim.Adagrad(params, lr=lr,
                                        weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)

    return optimizer

if __name__ == '__main__':
    pass


