import torch
import torch.nn as nn
from dataclasses import dataclass

from simple_parsing import ArgumentParser, choice
import copy
from Methods.models.utils import misc
from Methods.models.utils.PoolGateHyper_model import HyperNetwork


device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HyperGate_Net(nn.Module):
    @dataclass
    class Options:
        #structural learning rate
        lr_structural: float = 0.001
        lr=0.01
        gate_dropout_rate = -1
        gate_arch= [10,10]
        module_type: str = choice('resnet','mlp','zenke',default='mlp')

    def __init__(self, options:Options=Options(),num_tasks: int =6,temb_size = None, importance: int =1000,**kwargs):
        super().__init__()
        self.importance = importance
        self.params = None
        self.args: HyperGate_Net.Options = copy.copy(options)
        hyper_chunks = self.args.hyper_chunks
        assert(len(hyper_chunks) in [1,2,3])
        if len(hyper_chunks) == 1:
            hyper_chunks = hyper_chunks[0]

        hnet_arch = self.args.hnet_arch
        hnet_act = self.args.hnet_act
        if hnet_act is not None:
            hnet_act = misc.str_to_act(hnet_act)
        activation_fn_out = self.args.net_act_output_layer
        if activation_fn_out is not None:
            activation_fn_out = misc.str_to_act(activation_fn_out)
        use_bias = True
        no_weights=False
        hnet_dropout_rate = self.args.hnet_dropout_rate
        temb_std = self.args.temb_std
        self.PoolGateNet = HyperNetwork([[hyper_chunks]], num_tasks, verbose=False, layers=hnet_arch, activation_fn_out = activation_fn_out,
                                       te_dim=temb_size, activation_fn=hnet_act, use_bias=use_bias, no_weights=no_weights,
                                      init_weights=None, dropout_rate=hnet_dropout_rate, noise_dim=-1, temb_std=temb_std)
        self.fisher = self._init_fisher()

    def _init_fisher(self, task_id = 0):
        fisher = {}
        task_id_str = str(task_id)
        for n, p in self.PoolGateNet.named_parameters():
            if not 'task_embs' in n:
                if p.requires_grad:
                    fisher[n] = torch.zeros_like(p.data)
            else:
                if task_id_str in n:
                    if p.requires_grad:
                        fisher[n] = torch.zeros_like(p.data)
        fisher_cuda = {k: v.to("cuda") for k, v in fisher.items()}
        return fisher_cuda


    def _compute_fisher(self, projection_phase_length, task_id):
        task_id_str = str(task_id)
        for n, p in self.PoolGateNet.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if not 'task_embs' in n:
                        self.fisher[n] += (p.grad ** 2) / projection_phase_length
                    elif task_id_str in n:
                        self.fisher[n] += (p.grad ** 2) / projection_phase_length

        return self.fisher

    def penalty(self, task_id):
        task_id_str = str(task_id)
        loss = 0
        for n, p in self.PoolGateNet.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    if ('task_embs' not in n) or (task_id_str in n):
                        _loss = self.fisher[n] * (p - self.params[n]) ** 2
                        loss += _loss.sum()
        return loss * (self.importance / 2)


