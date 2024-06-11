
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Methods.models.utils import init_utils as iutils
from .module_wrappers import CLHyperNetInterface
from .torch_utils import init_params

class HyperNetwork(nn.Module, CLHyperNetInterface):

    def __init__(self, target_shapes, num_tasks, layers=[50, 100], verbose=True,
                 te_dim=8, no_te_embs=False, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_weights=False, init_weights=None,
                 ce_dim=None, dropout_rate=-1, use_spectral_norm=False,
                 create_feedback_matrix=False, target_net_out_dim=10,
                 random_scale_feedback_matrix = 1., activation_fn_out=torch.nn.ReLU(),
                 use_batch_norm=False, noise_dim=-1, temb_std=-1):


        nn.Module.__init__(self)
        CLHyperNetInterface.__init__(self)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this hypernetwork type.')
        if use_batch_norm:

            if ce_dim is None:
                raise ValueError('Can\'t use batchnorm as long as ' +
                                 'hypernetwork process more than 1 sample ' +
                                 '("ce_dim" must be specified).')
            raise NotImplementedError('Batch normalization not yet ' +
                                      'implemented for this hypernetwork type.')

        assert(len(target_shapes) > 0)
        assert(no_te_embs or num_tasks > 0)
        self._num_tasks = num_tasks

        assert(init_weights is None or no_weights is False)
        self._no_weights = no_weights
        self._no_te_embs = no_te_embs
        self._te_dim = te_dim
        self._size_ext_input = ce_dim
        self._layers = layers
        self._target_shapes = target_shapes
        self._use_bias = use_bias
        self._act_fn = activation_fn
        self._act_fn_out = activation_fn_out
        self._init_weights = init_weights
        self._dropout_rate = dropout_rate
        self._noise_dim = noise_dim
        self._temb_std = temb_std
        self._shifts = None # FIXME temporary test.

        ### Hidden layers
        self._gen_layers(layers, te_dim, use_bias, no_weights, init_weights,
                         ce_dim, noise_dim)

        if create_feedback_matrix:
            self._create_feedback_matrix(target_shapes, target_net_out_dim,
                                         random_scale_feedback_matrix)

        self._dropout = None
        if dropout_rate != -1:
            assert(dropout_rate >= 0 and dropout_rate <= 1)
            self._dropout = nn.Dropout(dropout_rate)

        # Task embeddings.
        if no_te_embs:
            self._task_embs = None
        else:
            self._task_embs = nn.ParameterList()
            for _ in range(num_tasks):
                self._task_embs.append(nn.Parameter(data=torch.Tensor(te_dim),
                                                    requires_grad=True))
                torch.nn.init.normal_(self._task_embs[-1], mean=0., std=1.)

        self._theta_shapes = self._hidden_dims + self._out_dims

        ntheta = self.shapes_to_num_weights(self._theta_shapes)
        ntembs = int(np.sum([t.numel() for t in self._task_embs])) \
                if not no_te_embs else 0
        self._num_weights = ntheta + ntembs

        self._num_outputs = self.shapes_to_num_weights( \
            self.target_shapes)

        if verbose:
            print('Constructed hypernetwork with %d parameters (' % (ntheta \
                  + ntembs) + '%d network weights + %d task embedding weights).'
                  % (ntheta, ntembs))
            print('The hypernetwork has a total of %d outputs.'
                  % self._num_outputs)

        self._is_properly_setup()

    def _create_feedback_matrix(self, target_shapes, target_net_out_dim, 
                                random_scale_feedback_matrix):

        s = random_scale_feedback_matrix
        self._feedback_matrix = []
        for k in target_shapes:
            dims =  [target_net_out_dim] + k    
            self._feedback_matrix.append(torch.empty(dims).uniform_(-s, s))

    @property
    def feedback_matrix(self):

        return self._feedback_matrix

    def _gen_layers(self, layers, te_dim, use_bias, no_weights, init_weights,
                    ce_dim, noise_dim):

        ### Compute the shapes of all parameters.
        # Hidden layers.
        self._hidden_dims = []
        prev_dim = te_dim
        if ce_dim is not None:
            prev_dim += ce_dim
        if noise_dim != -1:
            prev_dim += noise_dim
        for i, size in enumerate(layers):
            self._hidden_dims.append([size, prev_dim])
            if use_bias:
                self._hidden_dims.append([size])
            prev_dim = size
        self._last_hidden_size = prev_dim

        # Output layers.
        self._out_dims = []
        for i, dims in enumerate(self.target_shapes):
            nouts = np.prod(dims)
            self._out_dims.append([nouts, self._last_hidden_size])
            if use_bias:
                self._out_dims.append([nouts])
        if no_weights:
            self._theta = None
            return

        self._theta = nn.ParameterList()
        for i, dims in enumerate(self._hidden_dims + self._out_dims):
            self._theta.append(nn.Parameter(torch.Tensor(*dims),
                                            requires_grad=True))

        if init_weights is not None:
            assert (len(init_weights) == len(self._theta))
            for i in range(len(init_weights)):
                assert (np.all(np.equal(list(init_weights[i].shape),
                                        list(self._theta[i].shape))))
                self._theta[i].data = init_weights[i]
        else:
            for i in range(0, len(self._theta), 2 if use_bias else 1):
                if use_bias:
                    init_params(self._theta[i], self._theta[i + 1])
                else:
                    init_params(self._theta[i])


    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True):

        if task_id is None and task_emb is None:
            raise Exception('The hyper network has to get either a task ID' +
                            'to choose the learned embedding or directly ' +
                            'get an embedding as input (e.g. from a task ' +
                            'recognition model).')

        if not self.has_theta and theta is None:
            raise Exception('Network was generated without internal weights. ' +
                            'Hence, "theta" option may not be None.')

        if theta is None:
            theta = self.theta
        else:
            assert(len(theta) == len(self.theta_shapes))
            for i, s in enumerate(self.theta_shapes):
                assert(np.all(np.equal(s, list(theta[i].shape))))

        if dTheta is not None:
            assert(len(dTheta) == len(self.theta_shapes))

            weights = []
            for i, t in enumerate(theta):
                weights.append(t + dTheta[i])
        else:
            weights = theta

        # Select task embeddings.
        if not self.has_task_embs and task_emb is None:
            raise Exception('The network was created with no internal task ' +
                            'embeddings, thus parameter "task_emb" has to ' +
                            'be specified.')

        if task_emb is None:
            task_emb = self._task_embs[task_id]
        if self.training and self._temb_std != -1:
            task_emb.add(torch.randn_like(task_emb) * self._temb_std)

        # Concatenate additional embeddings to task embedding, if given.
        if self.requires_ext_input and ext_inputs is None:
            raise Exception('The network was created to expect additional ' +
                            'inputs, thus parameter "ext_inputs" has to ' +
                            'be specified.')
        elif not self.requires_ext_input and ext_inputs is not None:
            raise Exception('The network was created to not expect ' +
                            'additional embeddings, thus parameter ' +
                            '"ext_inputs" cannot be specified.')

        if ext_inputs is not None:

            batch_size = ext_inputs.shape[0]
            task_emb = task_emb.expand(batch_size, self._te_dim)
            h = torch.cat([task_emb, ext_inputs], dim=1)
        else:
            batch_size = 1
            h = task_emb.expand(batch_size, self._te_dim)

        if self._noise_dim != -1:
            if self.training:
                eps = torch.randn((batch_size, self._noise_dim))
            else:
                eps = torch.zeros((batch_size, self._noise_dim))
            if h.is_cuda:
                eps = eps.to(h.get_device())
            h = torch.cat([h, eps], dim=1)

        # Hidden activations.
        for i in range(0, len(self._hidden_dims), 2 if self._use_bias else 1):
            b = None
            if self._use_bias:
                b = weights[i+1]
            h = F.linear(h, weights[i], bias=b)
            if self._act_fn is not None:
                h = self._act_fn(h)
            if self._dropout is not None:
                h = self._dropout(h)
        outputs = []
        j = 0
        for i in range(len(self._hidden_dims), len(self._theta_shapes),
                       2 if self._use_bias else 1):
            b = None
            if self._use_bias:
                b = weights[i+1]
            W = F.linear(h, weights[i], bias=b)
            if self._act_fn_out is not None:
                W = self._act_fn_out(W)
            W = W.view(batch_size, *self.target_shapes[j])
            if squeeze:
                W = torch.squeeze(W, dim=0)
            if self._shifts is not None: # FIXME temporary test!
                W += self._shifts[j]
            outputs.append(W)
            j += 1

        return outputs

    def apply_hyperfan_init(self, method='in', use_xavier=False,
                            temb_var=1., ext_inp_var=1.):

        if method not in ['in', 'out', 'harmonic']:
            raise ValueError('Invalid value for argument "method".')
        if not self.has_theta:
            raise ValueError('Hypernet without internal weights can\'t be ' +
                             'initialized.')

        ### Compute input variance ###
        if self._temb_std != -1:
            # Sum of uncorrelated variables.
            temb_var += self._temb_std**2

        assert self._size_ext_input is None or self._size_ext_input > 0
        assert self._noise_dim == -1 or self._noise_dim > 0

        inp_dim = self._te_dim + \
            (self._size_ext_input if self._size_ext_input is not None else 0) \
            + (self._noise_dim if self._noise_dim != -1 else 0)

        input_variance = (self._te_dim / inp_dim) * temb_var
        if self._size_ext_input is not None:
            input_variance += (self._size_ext_input / inp_dim) * ext_inp_var
        if self._noise_dim != -1:
            input_variance += (self._noise_dim / inp_dim) * 1.


        for i in range(0, len(self._hidden_dims), 2 if self._use_bias else 1):
            #W = self.theta[i]
            if use_xavier:
                iutils.xavier_fan_in_(self.theta[i])
            else:
                torch.nn.init.kaiming_uniform_(self.theta[i], mode='fan_in',
                                               nonlinearity='relu')

            if self._use_bias:
                #b = self.theta[i+1]
                torch.nn.init.constant_(self.theta[i+1], 0)

        ### Initialize output heads ###
        c_relu = 1 if use_xavier else 2

        c_bias = 1
        for s in self.target_shapes:
            if len(s) == 1:
                c_bias = 2
                break
        # This is how we should do it instead.
        #c_bias = 2 if mnet.has_bias else 1

        j = 0
        for i in range(len(self._hidden_dims), len(self._theta_shapes),
                       2 if self._use_bias else 1):

            if self._use_bias:
                #b = self.theta[i+1]
                torch.nn.init.constant_(self.theta[i+1], 0)

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out( \
                self.theta[i])

            out_shape = self.target_shapes[j]

            # FIXME 1D output tensors don't need to be bias vectors. They can
            # be arbitrary embeddings or, for instance, batchnorm weights.
            if len(out_shape) == 1: # Assume output is bias vector.
                m_fan_out = out_shape[0]

                if j > 0 and len(self.target_shapes[j-1]) > 1:
                    m_fan_in, _ = iutils.calc_fan_in_and_out( \
                        self.target_shapes[j-1])
                else:
                    # FIXME Quick-fix.
                    m_fan_in = m_fan_out

                var_in = c_relu / (2. * fan_in * input_variance)
                num = c_relu * (1. - m_fan_in/m_fan_out)
                denom = fan_in * input_variance
                var_out = max(0, num / denom)

            else:
                m_fan_in, m_fan_out = iutils.calc_fan_in_and_out(out_shape)

                var_in = c_relu / (c_bias * m_fan_in * fan_in * input_variance)
                var_out = c_relu / (m_fan_out * fan_in * input_variance)

            if method == 'in':
                var = var_in
            elif method == 'out':
                var = var_out
            elif method == 'harmonic':
                var = 2 * (1./var_in + 1./var_out)
            else:
                raise ValueError('Method %s invalid.' % method)

            # Initialize output head weight tensor using `var`.
            std = math.sqrt(var)
            a = math.sqrt(3.0) * std
            torch.nn.init._no_grad_uniform_(self.theta[i], -a, a)
            
            j += 1

    @staticmethod
    def shapes_to_num_weights(dims):

        return np.sum([np.prod(l) for l in dims])

if __name__ == '__main__':
    pass
