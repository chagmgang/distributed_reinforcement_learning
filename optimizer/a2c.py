import tensorflow as tf

def compute_entropy_loss(policy):
    log_policy = tf.log(policy)
    entropy = tf.reduce_sum(-policy * log_policy, axis=1)
    entropy = -tf.reduce_mean(entropy)
    return entropy 

def compute_baseline_loss(value, next_value,
                          discounts, reward):
    next_value = tf.stop_gradient(next_value)
    value_diff = reward + discounts * next_value - value
    value_loss = tf.reduce_mean(value_diff * value_diff)
    return value_loss

def compute_policy_loss(policy, action, value,
                        next_value, discounts,
                        reward, num_action):
    onehot_action = tf.one_hot(action, num_action)
    selected_policy = tf.reduce_sum(policy * onehot_action, axis=1)
    
    advantage = reward + discounts * next_value - value
    advantage = tf.stop_gradient(advantage)

    policy_loss = -tf.reduce_mean(advantage * selected_policy)
    return policy_loss
