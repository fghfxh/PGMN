
import matplotlib
import matplotlib.pyplot as plt
import math
from torch import nn
import torch
from warnings import warn

def init_params(weights, bias=None):

    warn('Function is deprecated. Use "utils.torch_utils.init_params" instead.',
         DeprecationWarning)

    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

def str_to_ints(str_arg):

    if isinstance(str_arg, int):
        return [str_arg]

    if len(str_arg) > 0:
        return [int(s) for s in str_arg.split(',')]
    else:
        return []

def list_to_str(list_arg, delim=' '):

    ret = ''
    for i, e in enumerate(list_arg):
        if i > 0:
            ret += delim
        ret += str(e)
    return ret

def str_to_act(act_str):

    if act_str == 'linear':
        act = None
    elif act_str == 'sigmoid':
        act = torch.nn.Sigmoid()
    elif act_str == 'relu':
        act = torch.nn.ReLU()
    elif act_str == 'elu':
        act = torch.nn.ELU()
    elif act_str == 'leakyrelu':
        act = torch.nn.LeakyReLU()
    else:
        raise Exception('Activation function %s unknown.' % act_str)
    return act

def configure_matplotlib_params(fig_size = [6.4, 4.8], two_axes=True,
                                font_size=8):

    params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'font.sans-serif': ['Arial'],
        'text.usetex': True,
        'text.latex.preamble': [r'\usepackage[scaled]{helvet}',
                                r'\usepackage{sfmath}'],
        'font.family': 'sans-serif',
        'legend.fontsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'axes.titlesize': font_size,
        'axes.spines.right' : not two_axes,
        'axes.spines.top' : not two_axes,
        'figure.figsize': fig_size,
        'legend.handlelength': 0.5
    }

    matplotlib.rcParams.update(params)

def get_colorbrewer2_colors(family = 'Set2'):

    if family == 'Set2':
        return [
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            '#ffff33',
            '#a65628',
            '#b3de69'
        ]
    if family == 'Set3':
        return [
            '#8dd3c7',
            '#ffffb3',
            '#bebada',
            '#fb8072',
            '#80b1d3',
            '#fdb462',
            ''
        ]
    elif family == 'Dark2':
        return [
            '#1b9e77',
            '#d95f02',
            '#7570b3',
            '#e7298a',
            '#66a61e',
            '#e6ab02',
            '#a6761d'
        ]
    elif family == 'Pastel':
        return [
            '#fbb4ae',
            '#b3cde3',
            '#ccebc5',
            '#decbe4',
            '#fed9a6',
            '#ffffcc',
            '#e5d8bd'
        ]

def repair_canvas_and_show_fig(fig, close=True):

    tmp_fig = plt.figure()
    tmp_manager = tmp_fig.canvas.manager
    tmp_manager.canvas.figure = fig
    fig.set_canvas(tmp_manager.canvas)
    plt.close(tmp_fig.number)
    plt.figure(fig.number)
    plt.show()
    if close:
        plt.close(fig.number)

if __name__ == '__main__':
    pass
