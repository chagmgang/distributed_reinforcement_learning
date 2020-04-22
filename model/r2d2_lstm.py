import tensorflow as tf
import numpy as np

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

def network(state, previous_action, initial_h, initial_c, lstm_size, num_action):
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)

    previous_action = tf.one_hot(previous_action, num_action)
    previous_action = tf.layers.dense(inputs=previous_action, units=256, activation=tf.nn.relu)
    previous_action = tf.layers.dense(inputs=previous_action, units=256, activation=tf.nn.relu)

    concat = tf.concat([state, previous_action], axis=1)
    expanded_state = tf.expand_dims(concat, axis=1)

    lstm_out, h, c = lstm(
        lstm_size=lstm_size,
        flatten=expanded_state,
        initial_h=initial_h,
        initial_c=initial_c)

    lstm_out = lstm_out[:, -1]
    q_value = tf.layers.dense(inputs=lstm_out, units=128, activation=tf.nn.relu)
    value = tf.layers.dense(inputs=q_value, units=num_action, activation=None)
    mean = tf.layers.dense(inputs=q_value, units=1, activation=None)
    return value-mean, h, c

def build_network(s_ph, h_ph, c_ph, pa_ph,
                  main_s_ph, main_h_ph, main_c_ph, main_d_ph, main_pa_ph,
                  target_s_ph, target_h_ph, target_c_ph, target_d_ph, target_pa_ph,
                  lstm_size, num_action):

    seq_len = main_s_ph.shape[1].value

    with tf.variable_scope('main'):
        q_value, h, c = network(
            state=s_ph,
            previous_action=pa_ph,
            initial_h=h_ph,
            initial_c=c_ph,
            lstm_size=lstm_size,
            num_action=num_action)

    main_q_value_stack = []
    main_h, main_c = main_h_ph, main_c_ph
    for i in range(seq_len):
        with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
            main_q_value, main_h, main_c = network(
                state=main_s_ph[:, i],
                previous_action=main_pa_ph[:, i],
                initial_h=main_h,
                initial_c=main_c,
                lstm_size=lstm_size,
                num_action=num_action)

            main_q_value_stack.append(main_q_value)
            expanded_main_done = tf.expand_dims(main_d_ph[:, i], axis=1)
            main_h = tf.to_float(~expanded_main_done) * main_h
            main_c = tf.to_float(~expanded_main_done) * main_c

    main_q_value_stack = tf.stack(main_q_value_stack)
    main_q_value_stack = tf.transpose(main_q_value_stack, perm=[1, 0, 2])

    with tf.variable_scope('target'):
        _, _, _ = network(
            state=s_ph,
            previous_action=pa_ph,
            initial_h=h_ph,
            initial_c=c_ph,
            lstm_size=lstm_size,
            num_action=num_action)

    target_q_value_stack = []
    target_h, target_c = target_h_ph, target_c_ph
    for i in range(seq_len):
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            target_q_value, target_h, target_c = network(
                state=target_s_ph[:, i],
                previous_action=target_pa_ph[:, i],
                initial_h=target_h,
                initial_c=target_c,
                lstm_size=lstm_size,
                num_action=num_action)

            target_q_value_stack.append(target_q_value)
            expanded_target_done = tf.expand_dims(target_d_ph[:, i], axis=1)
            target_h = tf.to_float(~expanded_target_done) * target_h
            target_c = tf.to_float(~expanded_target_done) * target_c

    target_q_value_stack = tf.stack(target_q_value_stack)
    target_q_value_stack = tf.transpose(target_q_value_stack, perm=[1, 0, 2])

    return q_value, h, c, main_q_value_stack, target_q_value_stack
