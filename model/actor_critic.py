import tensorflow as tf

def attention_CNN(x):
    x = tf.layers.conv2d(
            inputs=x, filters=32, kernel_size=[8, 8],
            strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(
            inputs=x, filters=64, kernel_size=[4, 4],
            strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(
            inputs=x, filters=64, kernel_size=[3, 3],
            strides=[1, 1], padding='VALID', activation=tf.nn.relu)
    shape = x.get_shape()
    return tf.layers.flatten(x), [s.value for s in shape]

def action_embedding(previous_action, num_action):
    onehot_action = tf.one_hot(previous_action, num_action)
    x = tf.layers.dense(inputs=onehot_action, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    return x

def fully_connected(x, hidden_list, output_size, final_activation):
    for h in hidden_list:
        x = tf.layers.dense(inputs=x, units=h, activation=tf.nn.relu)
    return tf.layers.dense(inputs=x, units=output_size, activation=final_activation)

def network(image, previous_action, num_action):

    image_embedding, _ = attention_CNN(image)
    previous_action_embedding = action_embedding(previous_action, num_action)
    concat = tf.concat([image_embedding, previous_action_embedding], axis=1)

    actor = fully_connected(
            concat, [256, 256], num_action, tf.nn.softmax)
    critic = tf.squeeze(fully_connected(
            concat, [256, 256], 1, None), axis=1)

    return actor, critic

def build_network(state, next_state,
                  previous_action, next_previous_action,
                  num_action):

    with tf.variable_scope('a3c'):
        policy, value = network(
                image=state,
                previous_action=previous_action,
                num_action=num_action)

    with tf.variable_scope('a3c', reuse=tf.AUTO_REUSE):
        next_policy, next_value = network(
                image=next_state,
                previous_action=next_previous_action,
                num_action=num_action)

    return policy, value, next_policy, next_value
