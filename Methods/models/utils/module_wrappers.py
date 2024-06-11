
from abc import ABC, abstractmethod
import numpy as np
from warnings import warn

class CLHyperNetInterface(ABC):

    def __init__(self):

        super(CLHyperNetInterface, self).__init__()

        warn('Please use class "hnets.hnet_interface.CLHyperNetInterface" ' +
             'instead.', DeprecationWarning)

        self._theta = None
        self._task_embs = None
        self._theta_shapes = None

        self._num_weights = None
        self._num_outputs = None

        self._size_ext_input = None
        self._target_shapes = None

    def _is_properly_setup(self):

        assert(self._theta_shapes is not None)
        assert(self._num_weights is not None)
        assert(self._num_outputs is not None)
        assert(self._target_shapes is not None)

    @property
    def theta(self):

        return self._theta

    @property
    def num_outputs(self):

        return self._num_outputs

    @property
    def num_weights(self):

        return self._num_weights

    @property
    def has_theta(self):

        return self._theta is not None

    @property
    def theta_shapes(self):

        return self._theta_shapes

    @property
    def has_task_embs(self):

        return self._task_embs is not None

    @property
    def num_task_embs(self):

        assert(self.has_task_embs)
        return len(self._task_embs)

    @property
    def requires_ext_input(self):

        return self._size_ext_input is not None

    @property
    def target_shapes(self):

        return self._target_shapes

    def get_task_embs(self):

        assert(self.has_task_embs)
        return self._task_embs

    def get_task_emb(self, task_id):

        assert(self.has_task_embs)
        return self._task_embs[task_id]

    @abstractmethod
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True):

        pass # TODO implement

class MainNetInterface(ABC):

    def __init__(self):

        super(MainNetInterface, self).__init__()

        warn('Please use class "mnets.mnet_interface.MainNetInterface" ' +
             'instead.', DeprecationWarning)

        self._weights = None
        self._all_shapes = None
        self._hyper_shapes = None
        self._num_params = None
        self._has_bias = None
        self._has_fc_out = None

    def _is_properly_setup(self):

        assert(self._weights is not None or self._hyper_shapes is not None)
        if self._weights is not None and self._hyper_shapes is not None:
            assert((len(self._weights) + len(self._hyper_shapes)) == \
                   len(self._all_shapes))
        elif self._weights is not None:
            assert(len(self._weights) == len(self._all_shapes))
        else:
            assert(len(self._hyper_shapes) == len(self._all_shapes))
        assert(self._all_shapes is not None)
        assert(isinstance(self._has_bias, bool))
        assert(isinstance(self._has_fc_out, bool))

    @property
    def weights(self):

        return self._weights

    @property
    def param_shapes(self):

        return self._all_shapes

    @property
    def hyper_shapes(self):

        return self._hyper_shapes

    @property
    def has_bias(self):

        return self._has_bias

    @property
    def has_fc_out(self):

        return self._has_fc_out

    @property
    def num_params(self):

        if self._num_params is None:
            self._num_params = int(np.sum([np.prod(l) for l in
                                           self.param_shapes]))
        return self._num_params

if __name__ == '__main__':
    pass


