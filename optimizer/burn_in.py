import tensorflow as tf

def slice_in_burnin(size, tensor):
    return tensor[:, size:]

def reformat_tensor(main_q_value, target_q_value, reward, action, done, mask):
    current_state_q_value = main_q_value[:, :-1]
    next_main_q_value = main_q_value[:, 1:]
    next_state_q_value = target_q_value[:, 1:]
    reward = reward[:, :-1]
    action = action[:, :-1]
    done = done[:, :-1]
    mask = mask[:, :-1]

    return current_state_q_value, next_main_q_value, next_state_q_value, reward, action, done, mask

def select_state_value_action(q_value, action, num_action):
    onehot_action = tf.one_hot(action, num_action)
    action_q_value = q_value * onehot_action
    action_q_value = tf.reduce_sum(action_q_value, axis=2)
    return action_q_value

def value_function_rescaling(x, eps):
  """Value function rescaling per R2D2 paper, table 2."""
  return tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1.) - 1.) + eps * x

def inverse_value_function_rescaling(x, eps):
  """See Proposition A.2 in paper "Observe and Look Further"."""
  return tf.math.sign(x) * (
      tf.math.square(((tf.math.sqrt(
          1. + 4. * eps * (tf.math.abs(x) + 1. + eps))) - 1.) / (2. * eps)) -
      1.)
