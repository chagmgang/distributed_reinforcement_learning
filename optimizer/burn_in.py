import tensorflow as tf

def slice_in_burnin(size, tensor):
    return tensor[:, size:]

def reformat_tensor(main_q_value, target_q_value, reward, action, done):
    current_state_q_value = main_q_value[:, :-1]
    next_main_q_value = main_q_value[:, 1:]
    next_state_q_value = target_q_value[:, 1:]
    reward = reward[:, :-1]
    action = action[:, :-1]
    done = done[:, :-1]

    return current_state_q_value, next_main_q_value, next_state_q_value, reward, action, done

def select_state_value_action(q_value, action, num_action):
    onehot_action = tf.one_hot(action, num_action)
    action_q_value = q_value * onehot_action
    action_q_value = tf.reduce_sum(action_q_value, axis=2)
    return action_q_value