from model import r2d2_lstm
from optimizer import burn_in

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
        self.burnin_size = int(self.trajectory / 2)

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

                self.trajectory_reward = tf.placeholder(tf.float32, shape=[None, trajectory])
                self.trajectory_action = tf.placeholder(tf.int32, shape=[None, trajectory])
                self.trajectory_done = tf.placeholder(tf.bool, shape=[None, trajectory])

                if reward_clipping == 'abs_one':
                    self.trajectory_reward = tf.clip_by_value(self.trajectory_reward, -1.0, 1.0)
                elif reward_clipping == 'soft_asymmetric':
                    squeezed = tf.tanh(self.trajectory_reward / 5.0)
                    self.trajectory_reward = tf.where(self.trajectory_reward < 0, .3 * squeezed, squeezed) * 5.

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

                self.burn_in_main_q_value = burn_in.slice_in_burnin(self.burnin_size, self.main_q_value)
                self.burn_in_target_q_value = burn_in.slice_in_burnin(self.burnin_size, self.target_q_value)
                self.burn_in_action = burn_in.slice_in_burnin(self.burnin_size, self.trajectory_action)
                self.burn_in_reward = burn_in.slice_in_burnin(self.burnin_size, self.trajectory_reward)
                self.burn_in_done = burn_in.slice_in_burnin(self.burnin_size, self.trajectory_done)

                self.state_value, self.next_state_value, self.reward, self.action, self.done = burn_in.reformat_tensor(
                    main_q_value=self.burn_in_main_q_value,
                    target_q_value=self.burn_in_target_q_value,
                    reward=self.burn_in_reward,
                    action=self.burn_in_action,
                    done=self.burn_in_done)

                self.next_action = tf.argmax(self.next_state_value, axis=2)

                self.state_action_value = burn_in.select_state_value_action(
                    q_value=self.state_value, action=self.action, num_action=self.num_action)
                self.next_state_action_value = burn_in.select_state_value_action(
                    q_value=self.next_state_value, action=self.next_action, num_action=self.num_action)

                self.target_value = tf.stop_gradient(self.next_state_action_value * tf.to_float(~self.done) * self.discount_factor + self.reward)
                
                self.value_loss = tf.reduce_mean((self.target_value - self.state_action_value) ** 2)
            
            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(self.start_learning_rate, self.num_env_frames, self.learning_frame, self.end_learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.value_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variable), global_step=self.num_env_frames)

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

    def train(self, state, action, reward, done, initial_h, initial_c, previous_action):
        state = np.stack(state)
        action = np.stack(action)
        reward = np.stack(reward)
        done = np.stack(done)
        initial_h = np.stack(initial_h)
        initial_c = np.stack(initial_c)
        previous_action = np.stack(previous_action)

        value_loss, _, learning_rate = self.sess.run(
            [self.value_loss, self.train_op, self.learning_rate],
            feed_dict={
                self.trajectory_main_state: state,
                self.trajectory_main_previous_action: previous_action,
                self.trajectory_main_initial_c: initial_c,
                self.trajectory_main_initial_h: initial_h,

                self.trajectory_target_state: state,
                self.trajectory_target_previous_action: previous_action,
                self.trajectory_target_initial_c: initial_c[:, 0],
                self.trajectory_target_initial_h: initial_h[:, 0],
                self.trajectory_target_done: done,

                self.trajectory_reward: reward,
                self.trajectory_action: action,
                self.trajectory_done: done})

        return value_loss, learning_rate

    # def test(self):
    #     self.target_to_main()
    #     pa = 0
    #     previous_h = np.zeros([1, self.lstm_hidden_size])
    #     previous_c = np.zeros([1, self.lstm_hidden_size])
    #     done = False

    #     state_list = []
    #     previous_action_list = []
    #     done_list = []
    #     previous_h_list = []
    #     previous_c_list = []
    #     reward_list = []

    #     q_value_list = []

    #     for i in range(self.trajectory):
    #         s = np.random.rand(*self.input_shape)

    #         q_value, h, c = self.sess.run(
    #             [self.one_step_q_value, self.one_step_h, self.one_step_c],
    #             feed_dict={
    #                 self.state: [s],
    #                 self.previous_action: [pa],
    #                 self.initial_h: previous_h,
    #                 self.initial_c: previous_c})

    #         q_value_list.append(q_value)

    #         action = q_value[0]
    #         action = np.argmax(action)

    #         if np.random.rand() > 0.5:
    #             done = True
    #             reward = 1
    #         else:
    #             done = False
    #             reward = 0

    #         state_list.append(s)
    #         previous_action_list.append(pa)
    #         done_list.append(done)
    #         reward_list.append(reward)
    #         previous_h_list.append(previous_h)
    #         previous_c_list.append(previous_c)

    #         pa = action
    #         previous_h = h
    #         previous_c = c

    #         if done:
    #             s = np.random.rand(*self.input_shape)
    #             pa = 0
    #             previous_h = np.zeros([1, self.lstm_hidden_size])
    #             previous_c = np.zeros([1, self.lstm_hidden_size])
    #             done = False
    #     q_value_list = np.stack(q_value_list)[:, 0]
    #     state_list = np.stack(state_list)
    #     previous_action_list = np.stack(previous_action_list)
    #     done_list = np.stack(done_list)
    #     previous_h_list = np.stack(previous_h_list)[:, 0]
    #     previous_c_list = np.stack(previous_c_list)[:, 0]
    #     m_q_value = self.sess.run(
    #         self.main_q_value,
    #         feed_dict={
    #             self.trajectory_main_state: [state_list],
    #             self.trajectory_main_previous_action: [previous_action_list],
    #             self.trajectory_main_initial_h: [previous_h_list],
    #             self.trajectory_main_initial_c: [previous_c_list]})
    #     m_q_value = m_q_value[0]

    #     t_q_value = self.sess.run(
    #         self.target_q_value,
    #         feed_dict={
    #             self.trajectory_target_state: [state_list],
    #             self.trajectory_target_previous_action: [previous_action_list],
    #             self.trajectory_target_initial_h: [previous_h_list[0]],
    #             self.trajectory_target_initial_c: [previous_c_list[0]],
    #             self.trajectory_target_done: [done_list]})

    #     t_q_value = t_q_value[0]

    #     self.train(
    #         state=[state_list],
    #         action=[previous_action_list],
    #         reward=[reward_list],
    #         done=[done_list],
    #         initial_h=[previous_h_list],
    #         initial_c=[previous_c_list],
    #         previous_action=[previous_action_list])


    #     print('------------')
    #     print(q_value_list - t_q_value)
    #     print('------------')
    #     print(q_value_list - m_q_value)
    #     print('------------')