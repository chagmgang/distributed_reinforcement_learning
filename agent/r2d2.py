from model import r2d2_lstm

import tensorflow as tf
import numpy as np

import json
import utils
import random

class Agent:

    def __init__(self, trajectory, input_shape, num_action, lstm_hidden_size,
                 discount_factor, start_learning_rate, end_learning_rate,
                 learning_frame, baseline_loss_coef, entropy_coef,
                 gradient_clip_norm, reward_clipping, model_name, learner_name):

        self.input_shape = input_shape
        self.trajectory = trajectory
        self.num_action = num_action
        self.lstm_hidden_size = lstm_hidden_size
        self.discount_factor = discount_factor
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_frame = learning_frame
        self.baseline_loss_coef = baseline_loss_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm

        with tf.variable_scope(model_name):
            with tf.device('cpu'):
                self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
                self.previous_action = tf.placeholder(tf.int32, shape=[None])
                self.initial_c = tf.placeholder(tf.float32, shape=[None, lstm_hidden_size])
                self.initial_h = tf.placeholder(tf.float32, shape=[None, lstm_hidden_size])

                self.trajectory_main_state = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
                self.trajectory_main_previous_action = tf.placeholder(tf.int32, shape=[None, trajectory])
                self.trajectory_main_initial_c = tf.placeholder(tf.float32, shape=[None, trajectory, lstm_hidden_size])
                self.trajectory_main_initial_h = tf.placeholder(tf.float32, shape=[None, trajectory, lstm_hidden_size])

                self.trajectory_target_state = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
                self.trajectory_target_previous_action = tf.placeholder(tf.int32, shape=[None, trajectory])
                self.trajectory_target_initial_c = tf.placeholder(tf.float32, shape=[None, lstm_hidden_size])
                self.trajectory_target_initial_h = tf.placeholder(tf.float32, shape=[None, lstm_hidden_size])
                self.trajectory_target_done = tf.placeholder(tf.bool, shape=[None, trajectory])

                self.one_step_q_value, self.one_step_h, self.one_step_c, \
                    self.main_q_value, self.target_q_value = r2d2_lstm.build_network(
                        state=self.state,
                        previous_action=self.previous_action,
                        initial_h=self.initial_h, initial_c=self.initial_c,
                        trajectory_main_state=self.trajectory_main_state,
                        trajectory_main_previous_action=self.trajectory_main_previous_action,
                        trajectory_main_initial_h=self.trajectory_main_initial_h,
                        trajectory_main_initial_c=self.trajectory_main_initial_c,
                        trajectory_target_state=self.trajectory_target_state,
                        trajectory_target_previous_action=self.trajectory_target_previous_action,
                        trajectory_target_initial_h=self.trajectory_target_initial_h,
                        trajectory_target_initial_c=self.trajectory_target_initial_c,
                        trajectory_target_done=self.trajectory_target_done,
                        lstm_size=self.lstm_hidden_size,
                        num_action=self.num_action,
                        hidden_list=[256, 256])

        self.main_target = utils.copy_src_to_dst(f'{learner_name}/target', f'{learner_name}/main')
        self.global_to_session = utils.copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def target_to_main(self):
        self.sess.run(self.main_target)

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def get_policy_and_action(self, state, previous_action, h, c, epsilon):
        normalized_state = np.stack(state) / 255
        q_value, c, h = self.sess.run(
            [self.one_step_q_value, self.one_step_c, self.one_step_h],
            feed_dict={
                self.state: [normalized_state],
                self.previous_action: [previous_action],
                self.initial_c: [c],
                self.initial_h: [h]})

        q_value = q_value[0]
        
        if np.random.rand() > epsilon:
            action = np.argmax(q_value)
        else:
            action = np.random.choice(self.num_action)

        return action, q_value, q_value[action], h[0], c[0]

    def test(self):
        self.target_to_main()
        pa = 0
        previous_h = np.zeros([1, self.lstm_hidden_size])
        previous_c = np.zeros([1, self.lstm_hidden_size])
        done = False

        state_list = []
        previous_action_list = []
        done_list = []
        previous_h_list = []
        previous_c_list = []

        q_value_list = []

        for i in range(self.trajectory):
            s = np.random.rand(*self.input_shape)

            q_value, h, c = self.sess.run(
                [self.one_step_q_value, self.one_step_h, self.one_step_c],
                feed_dict={
                    self.state: [s],
                    self.previous_action: [pa],
                    self.initial_h: previous_h,
                    self.initial_c: previous_c})

            q_value_list.append(q_value)

            action = q_value[0]
            action = np.argmax(action)

            if np.random.rand() > 0.5:
                done = True
            else:
                done = False

            state_list.append(s)
            previous_action_list.append(pa)
            done_list.append(done)
            previous_h_list.append(previous_h)
            previous_c_list.append(previous_c)

            pa = action
            previous_h = h
            previous_c = c

            if done:
                s = np.random.rand(*self.input_shape)
                pa = 0
                previous_h = np.zeros([1, self.lstm_hidden_size])
                previous_c = np.zeros([1, self.lstm_hidden_size])
                done = False
        q_value_list = np.stack(q_value_list)[:, 0]
        state_list = np.stack(state_list)
        previous_action_list = np.stack(previous_action_list)
        done_list = np.stack(done_list)
        previous_h_list = np.stack(previous_h_list)[:, 0]
        previous_c_list = np.stack(previous_c_list)[:, 0]
        m_q_value = self.sess.run(
            self.main_q_value,
            feed_dict={
                self.trajectory_main_state: [state_list],
                self.trajectory_main_previous_action: [previous_action_list],
                self.trajectory_main_initial_h: [previous_h_list],
                self.trajectory_main_initial_c: [previous_c_list]})
        m_q_value = m_q_value[0]

        t_q_value = self.sess.run(
            self.target_q_value,
            feed_dict={
                self.trajectory_target_state: [state_list],
                self.trajectory_target_previous_action: [previous_action_list],
                self.trajectory_target_initial_h: [previous_h_list[0]],
                self.trajectory_target_initial_c: [previous_c_list[0]],
                self.trajectory_target_done: [done_list]})

        t_q_value = t_q_value[0]


        print('------------')
        print(q_value_list - t_q_value)
        print('------------')
        print(q_value_list - m_q_value)
        print('------------')