import tensorflow as tf
import numpy as np

def cnn(x):
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    shape = x.get_shape()
    return tf.layers.flatten(x), [s.value for s in shape]

def action_embedding(previous_action, num_action):
    onehot_action = tf.one_hot(previous_action, num_action)
    x = tf.layers.dense(inputs=onehot_action, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    return x

def lstm(lstm_size, flatten, initial_h, initial_c):
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)
    cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    output, state = tf.nn.dynamic_rnn(
        cell, flatten, dtype=tf.float32,
        initial_state=initial_state)
    c, h = state
    
    return output, h, c

def fully_connected(x, hidden_list, output_size, final_activation):
    for step, h in enumerate(hidden_list):
        x = tf.layers.dense(inputs=x, units=h, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=output_size, activation=final_activation)
    return x

def network(state, previous_action, initial_h, initial_c, lstm_size, hidden_list, num_action):
    image_embedding, _ = cnn(
        x=state)
    previous_action_embedding = action_embedding(
        previous_action=previous_action,
        num_action=num_action)
    concat = tf.concat([image_embedding, previous_action_embedding], axis=1)
    expanded_concat = tf.expand_dims(concat, axis=1)
    lstm_embedding, h, c = lstm(
        lstm_size=lstm_size,
        flatten=expanded_concat,
        initial_h=initial_h,
        initial_c=initial_c)
    lstm_embedding = lstm_embedding[:, -1]
    q_value = fully_connected(
        x=lstm_embedding,
        hidden_list=hidden_list,
        output_size=num_action,
        final_activation=None)
    return q_value, h, c

def simple_network(state, previous_action, initial_h, initial_c, lstm_size, hidden_list, num_action):
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)
    previous_action_embedding = action_embedding(
        previous_action=previous_action, num_action=num_action)
    concat = tf.concat([state, previous_action_embedding], axis=1)
    expanded_concat = tf.expand_dims(concat, axis=1)
    lstm_embedding, h, c = lstm(
        lstm_size=lstm_size,
        flatten=expanded_concat,
        initial_h=initial_h,
        initial_c=initial_c)
    lstm_embedding = lstm_embedding[:, -1]
    q_value = fully_connected(
        x=lstm_embedding,
        hidden_list=hidden_list,
        output_size=num_action,
        final_activation=None)
    return q_value, h, c

def dueuling_simple_network(state, previous_action, initial_h, initial_c, lstm_size, hidden_list, num_action):
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)
    previous_action_embedding = action_embedding(
        previous_action=previous_action, num_action=num_action)
    concat = tf.concat([state, previous_action_embedding], axis=1)
    expanded_concat = tf.expand_dims(concat, axis=1)
    lstm_embedding, h, c = lstm(
        lstm_size=lstm_size,
        flatten=expanded_concat,
        initial_h=initial_h,
        initial_c=initial_c)
    lstm_embedding = lstm_embedding[:, -1]
    value = fully_connected(
        x=lstm_embedding,
        hidden_list=hidden_list,
        output_size=num_action,
        final_activation=None)
    mean = fully_connected(
        x=lstm_embedding,
        hidden_list=hidden_list,
        output_size=1,
        final_activation=None)
    return value - mean, h, c

def build_simple_network(state, previous_action, initial_h, initial_c,
                         
                         trajectory_main_state, trajectory_main_previous_action,
                         trajectory_main_initial_h, trajectory_main_initial_c,

                         trajectory_target_state, trajectory_target_previous_action,
                         trajectory_target_initial_h, trajectory_target_initial_c,
                         trajectory_target_done,
                         
                         lstm_size,
                         num_action,
                         hidden_list):

    seq_len = trajectory_target_state.shape[1].value

    with tf.variable_scope('main'):
        one_step_q_value, one_step_h, one_step_c = dueuling_simple_network(
            state=state, previous_action=previous_action,
            initial_h=initial_h, initial_c=initial_c,
            lstm_size=lstm_size, hidden_list=hidden_list, num_action=num_action)

    main_q_value_stack = []
    for i in range(seq_len):
        with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
            main_q_value, _, _ = dueuling_simple_network(
                state=trajectory_main_state[:, i],
                previous_action=trajectory_main_previous_action[:, i],
                initial_h=trajectory_main_initial_h[:, i],
                initial_c=trajectory_main_initial_c[:, i],
                lstm_size=lstm_size,
                hidden_list=hidden_list,
                num_action=num_action)
            main_q_value_stack.append(main_q_value)
    main_q_value_stack = tf.stack(main_q_value_stack)
    main_q_value_stack = tf.transpose(main_q_value_stack, perm=[1, 0, 2])

    with tf.variable_scope('target'):
        _, _, _ = dueuling_simple_network(
            state=state, previous_action=previous_action,
            initial_h=initial_h, initial_c=initial_c,
            lstm_size=lstm_size, hidden_list=hidden_list, num_action=num_action)
        
    target_q_value_stack = []
    target_h, target_c = trajectory_target_initial_h, trajectory_target_initial_c
    for i in range(seq_len):
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            target_q_value, target_h, target_c = dueuling_simple_network(
                state=trajectory_target_state[:, i],
                previous_action=trajectory_target_previous_action[:, i],
                initial_h=target_h,
                initial_c=target_c,
                lstm_size=lstm_size,
                hidden_list=hidden_list,
                num_action=num_action)

            expanded_done = tf.expand_dims(trajectory_target_done[:, i], axis=1)
            target_h = tf.to_float(~expanded_done) * target_h
            target_c = tf.to_float(~expanded_done) * target_c

            target_q_value_stack.append(target_q_value)

    target_q_value_stack = tf.stack(target_q_value_stack)
    target_q_value_stack = tf.transpose(target_q_value_stack, perm=[1, 0, 2])

    return one_step_q_value, one_step_h, one_step_c, main_q_value_stack, target_q_value_stack