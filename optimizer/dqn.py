import tensorflow as tf

def take_state_action_value(state_value, action, num_action):
    onehot_action = tf.one_hot(action, num_action)
    state_action_value = state_value * onehot_action
    state_action_value = tf.reduce_sum(state_action_value, axis=1)
    return state_action_value