
import tensorflow as tf
import numpy as np

def attention_CNN(x):
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

def lstm(lstm_hidden_size, flatten, initial_h, initial_c):
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)
    cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)
    output, state = tf.nn.dynamic_rnn(
        cell, flatten, dtype=tf.float32,
        initial_state=initial_state)
    c, h = state
    return output, c, h

def fully_connected(x, hidden_list, output_size, final_activation):
    for h in hidden_list:
        x = tf.layers.dense(inputs=x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(inputs=x, units=output_size, activation=final_activation)


def network(image, previous_action, initial_h, initial_c, num_action, lstm_hidden_size):
    image_embedding, _ = attention_CNN(image)
    previous_action_embedding = action_embedding(previous_action, num_action)
    concat = tf.concat([image_embedding, previous_action_embedding], axis=1)
    expand_concat = tf.expand_dims(concat, axis=1)
    lstm_embedding, c, h = lstm(lstm_hidden_size, expand_concat, initial_h, initial_c)
    last_lstm_embedding = lstm_embedding[:, -1]
    actor = fully_connected(last_lstm_embedding, [256, 256], num_action, tf.nn.softmax)
    critic = tf.squeeze(fully_connected(last_lstm_embedding, [256, 256], 1, None), axis=1)
    return actor, critic, c, h

def build_network(state, previous_action, initial_h, initial_c,
                  trajectory_state, trajectory_previous_action, 
                  trajectory_initial_h, trajectory_initial_c,
                  num_action, lstm_hidden_size, trajectory):

    with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
        policy, _, c, h = network(
            image=state, previous_action=previous_action,
            initial_h=initial_h, initial_c=initial_c,
            num_action=num_action, lstm_hidden_size=lstm_hidden_size)

    unrolled_first_state = trajectory_state[:, :-2]
    unrolled_middle_state = trajectory_state[:, 1:-1]
    unrolled_last_state = trajectory_state[:, 2:]

    unrolled_first_previous_action = trajectory_previous_action[:, :-2]
    unrolled_middle_previous_action = trajectory_previous_action[:, 1:-1]
    unrolled_last_previous_action = trajectory_previous_action[:, 2:]

    unrolled_first_initial_h = trajectory_initial_h[:, :-2]
    unrolled_middle_initial_h = trajectory_initial_h[:, 1:-1]
    unrolled_last_initial_h = trajectory_initial_h[:, 2:]

    unrolled_first_initial_c = trajectory_initial_c[:, :-2]
    unrolled_middle_initial_c = trajectory_initial_c[:, 1:-1]
    unrolled_last_initial_c = trajectory_initial_c[:, 2:]

    unrolled_first_policy = []
    unrolled_first_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_first_state[:, i],
                previous_action=unrolled_first_previous_action[:, i],
                initial_h=unrolled_first_initial_h[:, i],
                initial_c=unrolled_first_initial_c[:, i],
                num_action=num_action, lstm_hidden_size=lstm_hidden_size)
            unrolled_first_policy.append(p)
            unrolled_first_value.append(v)
    unrolled_first_policy = tf.stack(unrolled_first_policy, axis=1)
    unrolled_first_value = tf.stack(unrolled_first_value, axis=1)

    unrolled_middle_policy = []
    unrolled_middle_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_middle_state[:, i],
                previous_action=unrolled_middle_previous_action[:, i],
                initial_h=unrolled_middle_initial_h[:, i],
                initial_c=unrolled_middle_initial_c[:, i],
                num_action=num_action, lstm_hidden_size=lstm_hidden_size)
            unrolled_middle_policy.append(p)
            unrolled_middle_value.append(v)
    unrolled_middle_policy = tf.stack(unrolled_middle_policy, axis=1)
    unrolled_middle_value = tf.stack(unrolled_middle_value, axis=1)

    unrolled_last_policy = []
    unrolled_last_value = []
    for i in range(trajectory - 2):
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            p, v, _, _ = network(
                image=unrolled_last_state[:, i],
                previous_action=unrolled_last_previous_action[:, i],
                initial_h=unrolled_last_initial_h[:, i],
                initial_c=unrolled_last_initial_c[:, i],
                num_action=num_action, lstm_hidden_size=lstm_hidden_size)
            unrolled_last_policy.append(p)
            unrolled_last_value.append(v)
    unrolled_last_policy = tf.stack(unrolled_last_policy, axis=1)
    unrolled_last_value = tf.stack(unrolled_last_value, axis=1)

    return policy, c, h, unrolled_first_policy, unrolled_first_value, \
        unrolled_middle_policy, unrolled_middle_value, \
            unrolled_last_policy, unrolled_last_value