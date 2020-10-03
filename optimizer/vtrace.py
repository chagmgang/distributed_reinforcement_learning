import tensorflow as tf

def split_data(x):
    '''
    x =         [x_(t), x_(t+1), x_(t+2), ..., x_(n)] of shape [None, (n-t+1), ...]
    first_x =   [x_(t), x_(t+1), x_(t+2), ..., x_(n-2)] of shape [None, (n-t-1), ...]
    middle_x =  [x_(t+1), x_(t+2), x_(t+3), ..., x_(n-1)] of shape [None, (n-t-1), ...]
    last_x =    ]x_(t+2), x_(t+3), x_(t+4), ..., x_(n)] of shape [None, (n-t-1), ...]
    '''
    first_x = x[:, :-2]
    middle_x = x[:, 1:-1]
    last_x = x[:, 2:]

    return first_x, middle_x, last_x

def log_probs_from_softmax_and_actions(policy_softmax, actions, action_size):
    '''
    INPUT:
    policy_softmax =    [pi(a|x_0),        pi(a|x_1),        ..., pi(a|x_n)]
    actions =           [a_0,              a_1,              ..., a_n]
    OUTPUT:
    log_prob =          [log(pi(a_0|x_0)), log(pi(a_1|x_1)), ..., log(pi(a_n|x_n))]
    '''
    onehot_action = tf.one_hot(actions, action_size)
    selected_softmax = tf.reduce_sum(policy_softmax * onehot_action, axis=2)
    log_prob = tf.log(selected_softmax)
    return log_prob

def from_softmax(behavior_policy_softmax, target_policy_softmax, actions, discounts,
                 rewards, values, next_values, action_size, clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0):
    '''
    INPUT:
    behavior_policy_softmax =   [mu(a|x_{0}), mu(a|x_{1}), ..., mu(a|x_{n})]        shape = [B, T, OUTPUT_SIZE]
    target_policy_softmax =     [pi(a|x_{0}), pi(a|x_{1}), ..., pi(a|x_{n})]        shape = [B, T, OUTPUT_SIZE]
    actions =                   [a_{0},       a_{1},     , ..., a_{n}]              shape = [B, T]
    discounts =                 [d_{0},       d_{1},     , ..., d_{n}]              shape = [B, T]
    rewards =                   [r_{0},       r_{1},     , ..., r_{n}]              shape = [B, T]
    values =                    [V(x_{0}),    V(x_{1}),  , ..., V(x_{n})]           shape = [B, T]
    next_values =               [V(x_{1}),    V(x_{2}),  , ..., V(x_{n+1})]         shape = [B, T]
    OUTPUT:
    vs =                        [vs_{0},      vs_{1},    , ..., vs_{n}]             shape = [B, T]
    clipped_rho =               [clipped_rho_{0}, clipped_rho_{1}, ..., clipped_rho_{n}]    shape = [B, T]
    '''

    target_action_log_probs = log_probs_from_softmax_and_actions(
        policy_softmax=target_policy_softmax, actions=actions, action_size=action_size)
    behavior_action_log_probs = log_probs_from_softmax_and_actions(
        policy_softmax=behavior_policy_softmax, actions=actions, action_size=action_size)

    log_rhos = target_action_log_probs - behavior_action_log_probs          # -> log( pi(a_s|x_s) / mu(a_s|x_s) )
    
    transpose_log_rhos = tf.transpose(log_rhos, perm=[1, 0])
    transpose_discounts = tf.transpose(discounts, perm=[1, 0])
    transpose_rewards = tf.transpose(rewards, perm=[1, 0])
    transpose_values = tf.transpose(values, perm=[1, 0])
    transpose_next_value = tf.transpose(next_values, perm=[1, 0])

    transpose_vs, transpose_clipped_rho = from_importance_weights(
        log_rhos=transpose_log_rhos, discounts=transpose_discounts,
        rewards=transpose_rewards, values=transpose_values,
        bootstrap_value=transpose_next_value[-1],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)
    
    vs = tf.transpose(transpose_vs, perm=[1, 0])
    clipped_rho = tf.transpose(transpose_clipped_rho, perm=[1, 0])

    return vs, clipped_rho

def from_importance_weights(log_rhos, discounts, rewards, values, bootstrap_value,
                            clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):

    rhos = tf.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos)
    else:
        clipped_rhos = rhos
    
    cs = tf.minimum(1.0, rhos, name='cs')
    values_t_plus_1 = tf.concat(
        [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    sequences = (discounts, cs, deltas)

    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = tf.zeros_like(bootstrap_value)
    vs_minus_v_xs = tf.scan(
        fn=scanfunc,
        elems=sequences,
        initializer=initial_values,
        parallel_iterations=1,
        back_prop=False,
        reverse=True,
        name='scan')
    vs = tf.add(vs_minus_v_xs, values)

    return tf.stop_gradient(vs), tf.stop_gradient(clipped_rhos)

def compute_policy_gradient_loss(softmax, actions, advantages, output_size):
    onehot_action = tf.one_hot(actions, output_size)
    selected_softmax = tf.reduce_sum(softmax * onehot_action, axis=2)
    selected_log_prob = tf.log(selected_softmax + 1e-8)
    advantages = tf.stop_gradient(advantages)
    # policy_gradient_loss_per_timestep = selected_log_prob[:, 0] * advantages[:, 0]
    policy_gradient_loss_per_timestep = selected_log_prob * advantages
    return -tf.reduce_sum(policy_gradient_loss_per_timestep)

def compute_baseline_loss(vs, value):
    # error = tf.stop_gradient(vs[:, 0]) - value[:, 0]
    error = tf.stop_gradient(vs) - value
    l2_loss = tf.square(error)
    return tf.reduce_sum(l2_loss) * 0.5

def compute_entropy_loss(softmax):
    policy = softmax
    log_policy = tf.log(softmax)
    entropy_per_time_step = -policy * log_policy
    # entropy_per_time_step = tf.reduce_sum(entropy_per_time_step[:, 0], axis=1)
    entropy_per_time_step = tf.reduce_sum(entropy_per_time_step, axis=1)
    return -tf.reduce_sum(entropy_per_time_step)
