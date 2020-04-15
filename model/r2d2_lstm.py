import tensorflow as tf
import numpy as np

def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

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
    
    return output, c, h

def fully_connected(x, hidden_list, output_size, final_activation):
    for step, h in enumerate(hidden_list):
        x = tf.layers.dense(inputs=x, units=h, activation=tf.nn.relu, name=f'fully_conntect_{step}')
    x = tf.layers.dense(inputs=x, units=output_size, activation=final_activation, name=f'fully_conntect_last')
    return x

def build_network(state,                            # main observation for inference
                  previous_action,                  # main previous_action for inference
                  initial_c,                        # main initial_c for inference
                  initial_h,                        # main initial_h for inference
                  
                  trajectory_main_state,            # main observation for training
                  trajectory_main_previous_action,  # main previous_action for training
                  trajectory_main_initial_c,        # main initial_c for training
                  trajectory_main_initial_h,        # main initial_h for training
                  
                  trajectory_target_state,          # target observation for training
                  trajectory_target_previous_action,# target previous_action for training
                  trajectory_target_initial_c,      # target initial_c for training
                  trajectory_target_initial_h,      # target initial_h for training

                  num_action,
                  lstm_size,
                  hidden_list):

    seq_len = trajectory_main_state.shape[1].value

    ### for one step infernece in main network
    with tf.variable_scope('main'):
        one_step_image_embedding, _ = cnn(state)
        one_step_previous_action_embedding = action_embedding(previous_action, num_action)
        one_step_concat = tf.concat([one_step_image_embedding, one_step_previous_action_embedding], axis=1)
        one_step_expanded_concat = tf.expand_dims(one_step_concat, axis=1)
        one_step_lstm_embedding, one_step_c, one_step_h = lstm(
            lstm_size=lstm_size,
            flatten=one_step_expanded_concat,
            initial_h=initial_h,
            initial_c=initial_c)
        one_step_lstm_embedding = one_step_lstm_embedding[:, -1]
        one_step_q_value = fully_connected(
            one_step_lstm_embedding, hidden_list, num_action, None)
            
    ### for seqeunce step training in main network
    main_training_cnn_stack = []
    main_training_previous_action_stack = []
    for i in range(seq_len):
        with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
            main_training_image_embedding, _ = cnn(trajectory_main_state[:, i])
            main_training_previous_action_embedding = action_embedding(trajectory_main_previous_action[:, i], num_action)
            main_training_cnn_stack.append(main_training_image_embedding)
            main_training_previous_action_stack.append(main_training_previous_action_embedding)

    main_training_cnn_stack = tf.stack(main_training_cnn_stack)
    main_training_previous_action_stack = tf.stack(main_training_previous_action_stack)
    main_training_cnn_stack = tf.transpose(main_training_cnn_stack, perm=[1, 0, 2])
    main_training_previous_action_stack = tf.transpose(main_training_previous_action_stack, perm=[1, 0, 2])
    main_training_concat = tf.concat([main_training_cnn_stack, main_training_previous_action_stack], axis=2)

    main_training_q_value_stack = []
    for i in range(seq_len):
        with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
            main_training_expanded_concat = tf.expand_dims(main_training_concat[:, i], axis=1)
            main_training_lstm_embedding, _, _ = lstm(
                lstm_size=lstm_size,
                flatten=main_training_expanded_concat,
                initial_h=trajectory_main_initial_h[:, i],
                initial_c=trajectory_main_initial_c[:, i])
            main_training_lstm_embedding = main_training_lstm_embedding[:, -1]
            main_training_q_value = fully_connected(
                main_training_lstm_embedding, hidden_list, num_action, None)
            main_training_q_value_stack.append(main_training_q_value)

    main_training_q_value_stack = tf.stack(main_training_q_value_stack)
    main_training_q_value_stack = tf.transpose(main_training_q_value_stack, perm=[1, 0, 2])

    ### for sequence step training in target network
    target_training_cnn_stack = []
    target_training_previous_action_stack = []
    for i in range(seq_len):
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            target_training_image_embedding, _ = cnn(trajectory_target_state[:, i])
            target_training_previous_action_embedding = action_embedding(
                trajectory_target_previous_action[:, i], num_action)
            target_training_cnn_stack.append(target_training_image_embedding)
            target_training_previous_action_stack.append(target_training_previous_action_embedding)

    target_training_cnn_stack = tf.stack(target_training_cnn_stack)
    target_training_previous_action_stack = tf.stack(target_training_previous_action_stack)
    target_training_cnn_stack = tf.transpose(target_training_cnn_stack, perm=[1, 0, 2])
    target_training_previous_action_stack = tf.transpose(target_training_previous_action_stack, perm=[1, 0, 2])
    target_training_concat = tf.concat([target_training_cnn_stack, target_training_previous_action_stack], axis=2)

    target_training_q_value_stack = []
    for i in range(seq_len):
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            target_training_expanded_concat = tf.expand_dims(target_training_concat[:, i], axis=1)
            target_training_lstm_embedding, trajectory_target_initial_c, trajectory_target_initial_h = lstm(
                lstm_size=lstm_size,
                flatten=target_training_expanded_concat,
                initial_h=trajectory_target_initial_h,
                initial_c=trajectory_target_initial_c)
            target_training_lstm_embedding = target_training_lstm_embedding[:, -1]
            target_training_q_value = fully_connected(
                target_training_lstm_embedding, hidden_list, num_action, None)
            target_training_q_value_stack.append(target_training_q_value)
    target_training_q_value_stack = tf.stack(target_training_q_value_stack)
    target_training_q_value_stack = tf.transpose(target_training_q_value_stack, perm=[1, 0, 2])
            
    return one_step_q_value, one_step_c, one_step_h, main_training_q_value_stack, target_training_q_value_stack


if __name__ == '__main__':

    trajectory = 3
    batch_size = 6
    num_action = 18
    lstm_size = 8
    hidden_list = [256, 256]

    input_shape = [84, 84, 4]

    state = tf.placeholder(tf.float32, shape=[None, *input_shape])
    previous_action = tf.placeholder(tf.int32, shape=[None])
    initial_c = tf.placeholder(tf.float32, shape=[None, lstm_size])
    initial_h = tf.placeholder(tf.float32, shape=[None, lstm_size])

    trajectory_main_state = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
    trajectory_main_previous_action = tf.placeholder(tf.int32, shape=[None, trajectory])
    trajectory_main_initial_c = tf.placeholder(tf.float32, shape=[None, trajectory, lstm_size])
    trajectory_main_initial_h = tf.placeholder(tf.float32, shape=[None, trajectory, lstm_size])

    trajectory_target_state = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
    trajectory_target_previous_action = tf.placeholder(tf.int32, shape=[None, trajectory])
    trajectory_target_initial_c = tf.placeholder(tf.float32, shape=[None, lstm_size])
    trajectory_target_initial_h = tf.placeholder(tf.float32, shape=[None, lstm_size])


    one_step_q_value, one_step_c, one_step_h, \
        main_training_q_value_stack, target_training_q_value_stack = \
            build_network(state=state,
                        previous_action=previous_action,
                        initial_c=initial_c,
                        initial_h=initial_h,
                        trajectory_main_state=trajectory_main_state,
                        trajectory_main_previous_action=trajectory_main_previous_action,
                        trajectory_main_initial_c=trajectory_main_initial_c,
                        trajectory_main_initial_h=trajectory_main_initial_h,
                        trajectory_target_state=trajectory_target_state,
                        trajectory_target_previous_action=trajectory_target_previous_action,
                        trajectory_target_initial_c=trajectory_target_initial_c,
                        trajectory_target_initial_h=trajectory_target_initial_h,
                        num_action=num_action,
                        lstm_size=lstm_size,
                        hidden_list=hidden_list)

    copy_dst = copy_src_to_dst(from_scope='target', to_scope='main')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    ### test main network that one step inference is same with model network output of trajectory data
    # state_numpy = np.ones([1, *input_shape])
    # previous_action_numpy = np.ones([1])
    # initial_c_numpy = np.ones([1, lstm_size])
    # initial_h_numpy = np.ones([1, lstm_size])

    # state_trajectory_numpy = np.ones([1, trajectory, *input_shape])
    # previous_action_trajectory_numpy = np.ones([1, trajectory])
    # initial_c_trajectory_numpy = np.ones([1, trajectory, lstm_size])
    # initial_h_trajectory_numpy = np.ones([1, trajectory, lstm_size])

    # a, b = sess.run(
    #     [one_step_q_value, main_training_q_value_stack],
    #     feed_dict={
    #         state: state_numpy,
    #         previous_action: previous_action_numpy,
    #         initial_c: initial_c_numpy,
    #         initial_h: initial_h_numpy,
    #         trajectory_main_state: state_trajectory_numpy,
    #         trajectory_main_previous_action: previous_action_trajectory_numpy,
    #         trajectory_main_initial_c: initial_c_trajectory_numpy,
    #         trajectory_main_initial_h: initial_h_trajectory_numpy})

    # print(a)
    # print(b)

    sess.run(copy_dst)

    ### test main network and target network have same output after parameter sync

    state_list = [np.ones([*input_shape]) * i for i in range(trajectory)]
    pa = 0
    previous_h = np.zeros([1, lstm_size])
    previous_c = np.zeros([1, lstm_size])

    h_list, c_list, previous_action_list, s_list = [], [], [], []

    for s in state_list:
        
        q_value, c, h = sess.run(
            [one_step_q_value, one_step_c, one_step_h],
            feed_dict={
                state: [s],
                previous_action: [pa],
                initial_c: previous_c,
                initial_h: previous_h})

        h_list.append(previous_h)
        c_list.append(previous_c)
        previous_action_list.append(pa)
        s_list.append(s)

        action = np.argmax(q_value, axis=1)[0]
        pa = action
        previous_c, previous_h = c, h


    h_list = np.stack(h_list)[:, 0]
    c_list = np.stack(c_list)[:, 0]
    previous_action_list = np.stack(previous_action_list)
    s_list = np.stack(s_list)

    h_list = np.stack([h_list])
    c_list = np.stack([c_list])
    previous_action_list = np.stack([previous_action_list])
    s_list = np.stack([s_list])

    a, b = sess.run(
        [main_training_q_value_stack, target_training_q_value_stack],
        feed_dict={
            trajectory_main_state: s_list,
            trajectory_main_previous_action: previous_action_list,
            trajectory_main_initial_c: c_list,
            trajectory_main_initial_h: h_list,
            trajectory_target_state: s_list,
            trajectory_target_previous_action: previous_action_list,
            trajectory_target_initial_h: h_list[:, 0],
            trajectory_target_initial_c: c_list[:, 0]
        }
    )

    print(a)
    print(b)