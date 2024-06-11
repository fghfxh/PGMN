
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .torch_utils import init_params

class SelfAttnLayer(nn.Module):

    def __init__(self, in_dim, use_spectral_norm):

        super(SelfAttnLayer,self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim ,
                                    out_channels=in_dim // 8, kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8,
                                  kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                    kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        if use_spectral_norm:
            self.query_conv = nn.utils.spectral_norm(self.query_conv)
            self.key_conv = nn.utils.spectral_norm(self.key_conv)
            self.value_conv = nn.utils.spectral_norm(self.value_conv)

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, ret_attention=False):

        m_batchsize, C, width, height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1, width*height).\
            permute(0,2,1)

        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height)
        energy =  torch.bmm(proj_query, proj_key) # f(x)^T g(x)

        attention = self.softmax(energy) # shape: B x N x N

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        if ret_attention:
            return out, attention
        return out

class SelfAttnLayerV2(nn.Module):

    def __init__(self, in_dim, use_spectral_norm, no_weights=False,
                 init_weights=None):

        super(SelfAttnLayerV2,self).__init__()
        assert(not no_weights or init_weights is None)
        if use_spectral_norm:
            raise NotImplementedError('Spectral norm not yet implemented ' +
                                      'for this layer type.')

        self.channel_in = in_dim

        self.softmax  = nn.Softmax(dim=-1)

        query_dim = [in_dim // 8, in_dim, 1, 1]

        key_dim = [in_dim // 8, in_dim, 1, 1]

        value_dim = [in_dim, in_dim, 1, 1]
        gamma_dim = [1]
        self._weight_shapes = [query_dim, [query_dim[0]],
                               key_dim, [key_dim[0]],
                               value_dim, [value_dim[0]],
                               gamma_dim
                              ]

        if no_weights:
            self._weights = None
            return

        ### Define and initialize network weights.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self._weight_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))

        if init_weights is not None:
            assert(len(init_weights) == len(self._weight_shapes))
            
            for i in range(len(init_weights)):
                assert(np.all(np.equal(list(init_weights[i].shape),
                                       list(self._weights[i].shape))))
                self._weights[i].data = init_weights[i]
        else:
            for i in range(0, len(self._weights)-1, 2):
                init_params(self._weights[i], self._weights[i+1])

            nn.init.constant_(self._weights[-1], 0)

    @property
    def weight_shapes(self):
        return self._weight_shapes

    @property
    def weights(self):
        return self._weights

    def forward(self, x, ret_attention=False, weights=None, dWeights=None):

        if self._weights is None and weights is None:
            raise Exception('Layer was generated without internal weights. ' +
                            'Hence, "weights" option may not be None.')

        if weights is None:
            weights = self.weights
        else:
            assert(len(weights) == len(self.weight_shapes))

        if dWeights is not None:
            assert(len(dWeights) == len(self.weight_shapes))

            new_weights = []
            for i, w in enumerate(weights):
                new_weights.append(w + dWeights[i])
            weights = new_weights

        m_batchsize, C, width, height = x.size()

        proj_query = F.conv2d(x, weights[0], bias=weights[1]). \
            view(m_batchsize,-1, width*height).permute(0,2,1)

        proj_key = F.conv2d(x, weights[2], bias=weights[3]). \
            view(m_batchsize, -1, width*height)
        energy =  torch.bmm(proj_query, proj_key) # f(x)^T g(x)

        attention = self.softmax(energy) # shape: B x N x N

        proj_value = F.conv2d(x, weights[4], bias=weights[5]). \
            view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = weights[6] * out + x

        if ret_attention:
            return out, attention
        return out

if __name__ == '__main__':
    pass


