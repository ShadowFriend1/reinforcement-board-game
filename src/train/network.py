import numpy as np
import tensorflow as tf
from tf_agents.networks import q_network


class RLQNetwork(q_network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 fc_layer_params=(75, 40),
                 mask_q_value=-(10 ** 5),
                 activation_fn=tf.keras.activations.relu,
                 name='RLQNetwork'):

        obs_spec = observation_spec['state']
        super(RLQNetwork, self).__init__(input_tensor_spec=observation_spec,
                                         state_spec=(),
                                         name=name)

        self._q_net = q_network.QNetwork(input_tensor_spec=obs_spec,
                                         action_spec=action_spec,
                                         fc_layer_params=fc_layer_params,
                                         activation_fn=activation_fn)

        self._mask_q_value = mask_q_value

    def call(self, observation, step_type=None, network_state=None):
        state = observation['state']
        mask = observation['mask']
        q_values, _ = self._q_net(state, step_type)

        small_constant = tf.constant(self._mask_q_value, dtype=q_values.dtype,
                                     shape=q_values.shape)
        zeros = tf.zeros(shape=mask.shape, dtype=mask.dtype)
        masked_q_values = tf.where(tf.math.equal(zeros, mask),
                                   small_constant, q_values)

        return masked_q_values, network_state
