import numpy as np
import tensorflow as tf
from tf_agents.networks import q_network, network


class MaskedNetwork(network.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 fc_layer_params=(75, 40),
                 mask_q_value=-(10 ** 5),
                 activation_fn=tf.keras.activations.relu,
                 name='MaskQNetwork'):
        obs_spec = observation_spec['state']
        super(MaskedNetwork, self).__init__(input_tensor_spec=observation_spec,
                                            state_spec=(),
                                            name=name)

        self._q_net = q_network.QNetwork(input_tensor_spec=obs_spec,
                                         action_spec=action_spec,
                                         fc_layer_params=fc_layer_params,
                                         activation_fn=activation_fn)

        self._q_net.create_variables()
        self._q_net.summary()

        self._mask_q_value = mask_q_value

    def call(self, observation, step_type=None, network_state=None):
        state = observation['state']
        q_values, _ = self._q_net(state, step_type)

        print(q_values)

        mask = observation['mask']

        print(mask)

        small_constant = tf.constant(self._mask_q_value, dtype=q_values.dtype,
                                     shape=q_values.shape)
        zeros = tf.zeros(shape=q_values.shape, dtype=mask.dtype)

        masked_q_values = tf.where(tf.math.equal(zeros, mask),
                                   small_constant, q_values)

        print(masked_q_values)

        return masked_q_values, network_state
