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

def fully_connected(x, hidden_list, num_action, final_activation):
    for h in hidden_list:
        x = tf.layers.dense(inputs=x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(inputs=x, units=num_action, activation=final_activation)

def dueling_network(image, previous_action, num_action, hidden_list):
    image_embedding, _ = attention_CNN(
        x=image)
    previous_action_embedding = action_embedding(
        previous_action=previous_action,
        num_action=num_action)
    concat = tf.concat([image_embedding, previous_action_embedding], axis=1)
    value = fully_connected(
        x=concat,
        hidden_list=hidden_list,
        num_action=num_action,
        final_activation=None)
    mean = fully_connected(
        x=concat,
        hidden_list=hidden_list,
        num_action=1,
        final_activation=None)
    q_value = value - mean
    return q_value

def build_network(current_state, next_state, previous_action, action, num_action, hidden_list):

    with tf.variable_scope('main'):
        main_q_value = dueling_network(
            image=current_state,
            previous_action=previous_action,
            num_action=num_action,
            hidden_list=hidden_list)

    with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
        next_main_q_value = dueling_network(
            image=next_state,
            previous_action=action,
            num_action=num_action,
            hidden_list=hidden_list)

    with tf.variable_scope('target'):
        target_q_value = dueling_network(
            image=next_state,
            previous_action=action,
            num_action=num_action,
            hidden_list=hidden_list)

    return main_q_value, next_main_q_value, target_q_value

def simple_network(state, previous_action, num_action):
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)
    state = tf.layers.dense(inputs=state, units=256, activation=tf.nn.relu)

    previous_action = tf.one_hot(previous_action, num_action)
    previous_action = tf.layers.dense(inputs=previous_action, units=256, activation=tf.nn.relu)
    previous_action = tf.layers.dense(inputs=previous_action, units=256, activation=tf.nn.relu)

    concat = tf.concat([state, previous_action], axis=1)
    concat = tf.layers.dense(inputs=concat, units=256, activation=tf.nn.relu)
    value = tf.layers.dense(inputs=concat, units=num_action, activation=None)
    mean = tf.layers.dense(inputs=concat, units=1, activation=None)

    return value - mean

def build_simple_network(current_state, next_state, previous_action, action, num_action):

    with tf.variable_scope('main'):
        main_q_value = simple_network(
            state=current_state,
            previous_action=previous_action,
            num_action=num_action)
    with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
        next_main_q_value = simple_network(
            state=next_state,
            previous_action=action,
            num_action=num_action)
    with tf.variable_scope('target'):
        target_q_value = simple_network(
            state=next_state,
            previous_action=action,
            num_action=num_action)

    return main_q_value, next_main_q_value, target_q_value

if __name__ == '__main__':
    state_ph = tf.placeholder(tf.float32, shape=[None, 4])
    previous_action_ph = tf.placeholder(tf.int32, shape=[None])
    