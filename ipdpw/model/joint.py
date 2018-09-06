"""Joint models.

Provides models for joining features from DNA and CpG model.
"""
from __future__ import division
from __future__ import print_function

import inspect

from keras import layers as kl
from keras import models as km
from keras import regularizers as kr
from keras.layers.merge import concatenate

from .utils import Model

from ..utils import get_from_module


class JointModel(Model):
    """Abstract class of a Joint model."""

    def __init__(self, *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.mode = 'concat'
        self.scope = 'joint'

    def _get_inputs_outputs(self, models):
        inputs = []
        outputs = []
        for model in models:
            inputs.extend(model.inputs)
            outputs.extend(model.outputs)
        return (inputs, outputs)

    def _build(self, models, layers=[]):
        for layer in layers:
            layer.name = '%s/%s' % (self.scope, layer.name)

        inputs, outputs = self._get_inputs_outputs(models)
        x = concatenate(outputs)
        for layer in layers:
            x = layer(x)

        model = km.Model(inputs, x, name=self.name)
        return model


class JointL0(JointModel):
    """Concatenates inputs without trainable layers.

    .. code::

        Parameters: 0
    """

    def __call__(self, models):
        return self._build(models)


class JointL1h512(JointModel):
    """One fully-connected layer with 512 units.

    .. code::

        Parameters: 524,000
        Specification: fc[512]
    """

    def __init__(self, nb_layer=1, nb_hidden=512, *args, **kwargs):
        super(JointL1h512, self).__init__(*args, **kwargs)
        self.nb_layer = nb_layer
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        layers = []
        for layer in range(self.nb_layer):
            kernel_regularizer = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
            layers.append(kl.Dense(self.nb_hidden,
                                   kernel_initializer=self.init,
                                   kernel_regularizer=kernel_regularizer))
            layers.append(kl.Activation('relu'))
            layers.append(kl.Dropout(self.dropout))

        return self._build(models, layers)
