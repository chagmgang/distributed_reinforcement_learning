from model import r2d2_lstm
from optimizer import burn_in
from distributed_queue import buffer_queue

import tensorflow as tf
import numpy as np

import gym
import utils

class Agent:
    def __init__(self, input_shape, num_action, seq_len, lstm_size, hidden_list,
                 learning_rate,
                 discount_factor, model_name, learner_name):

        self.input_shape = input_shape
        self.num_action = num_action
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.hidden_list = hidden_list
        self.burn_in = int(self.seq_len / 2)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        with tf.variable_scope(model_name):
            with tf.device('cpu'):

                self.s_ph = tf.placeholder(tf.float32, shape=[None, *input_shape])
                self.pa_ph = tf.placeholder(tf.int32, shape=[None])
                self.initial_h_ph = tf.placeholder(tf.float32, shape=[None, lstm_size])
                self.initial_c_ph = tf.placeholder(tf.float32, shape=[None, lstm_size])

                self.trajectory_main_s_ph = tf.placeholder(tf.float32, shape=[None, seq_len, *input_shape])
                self.trajectory_main_pa_ph = tf.placeholder(tf.int32, shape=[None, seq_len])
                self.trajectory_main_initial_h_ph = tf.placeholder(tf.float32, shape=[None, seq_len, lstm_size])
                self.trajectory_main_initial_c_ph = tf.placeholder(tf.float32, shape=[None, seq_len, lstm_size])

                self.trajectory_target_s_ph = tf.placeholder(tf.float32, shape=[None, seq_len, *input_shape])
                self.trajectory_target_pa_ph = tf.placeholder(tf.int32, shape=[None, seq_len])
                self.trajectory_target_initial_h_ph = tf.placeholder(tf.float32, shape=[None, lstm_size])
                self.trajectory_target_initial_c_ph = tf.placeholder(tf.float32, shape=[None, lstm_size])
                self.trajectory_target_done = tf.placeholder(tf.bool, shape=[None, seq_len])

                self.trajectory_reward = tf.placeholder(tf.float32, shape=[None, seq_len])
                self.trajectory_action = tf.placeholder(tf.int32, shape=[None, seq_len])
                self.trajectory_done = tf.placeholder(tf.bool, shape=[None, seq_len])
                self.trajectory_weight = tf.placeholder(tf.float32, shape=[None])
                self.trajectory_mask = tf.placeholder(tf.bool, shape=[None, seq_len])

                self.one_q_value, self.one_h, self.one_c, self.main_q_value, self.target_q_value = r2d2_lstm.build_simple_network(
                    state=self.s_ph, previous_action=self.pa_ph,
                    initial_h=self.initial_h_ph, initial_c=self.initial_c_ph,

                    trajectory_main_state=self.trajectory_main_s_ph,
                    trajectory_main_previous_action=self.trajectory_main_pa_ph,
                    trajectory_main_initial_h=self.trajectory_main_initial_h_ph,
                    trajectory_main_initial_c=self.trajectory_main_initial_c_ph,

                    trajectory_target_state=self.trajectory_target_s_ph,
                    trajectory_target_previous_action=self.trajectory_target_pa_ph,
                    trajectory_target_initial_h=self.trajectory_target_initial_h_ph,
                    trajectory_target_initial_c=self.trajectory_target_initial_c_ph,
                    trajectory_target_done=self.trajectory_target_done,

                    lstm_size=lstm_size,
                    num_action=num_action,
                    hidden_list=hidden_list)

                self.sliced_main_q_value = burn_in.slice_in_burnin(
                    size=self.burn_in, tensor=self.main_q_value)
                self.sliced_target_q_value = burn_in.slice_in_burnin(
                    size=self.burn_in, tensor=self.target_q_value)
                self.sliced_reward = burn_in.slice_in_burnin(
                    size=self.burn_in, tensor=self.trajectory_reward)
                self.sliced_done = burn_in.slice_in_burnin(
                    size=self.burn_in, tensor=self.trajectory_done)
                self.sliced_action = burn_in.slice_in_burnin(
                    size=self.burn_in, tensor=self.trajectory_action)
                self.sliced_mask = burn_in.slice_in_burnin(
                    size=self.burn_in, tensor=self.trajectory_mask)

                self.reformed_main_q_value, self.reformed_next_main_q_value, \
                    self.reformed_target_q_value, self.reformed_reward, \
                        self.reformed_action, self.reformed_done, \
                            self.reformed_mask = burn_in.reformat_tensor(
                                main_q_value=self.sliced_main_q_value,
                                target_q_value=self.sliced_target_q_value,
                                reward=self.sliced_reward,
                                action=self.sliced_action,
                                done=self.sliced_done,
                                mask=self.sliced_mask)

                self.discounts = self.discount_factor * tf.to_float(~self.reformed_done)
                self.next_action = tf.argmax(self.reformed_next_main_q_value, axis=2)
                self.next_state_action_value = burn_in.select_state_value_action(
                    q_value=self.reformed_target_q_value, action=self.next_action, num_action=self.num_action)
                self.inverse_rescaled_next_state_action_value = burn_in.inverse_value_function_rescaling(
                    x=self.next_state_action_value, eps=1e-3)
                self.inverse_rescaled_target_value = self.discounts * self.inverse_rescaled_next_state_action_value + self.reformed_reward
                self.target_value = tf.stop_gradient(burn_in.value_function_rescaling(
                    x=self.inverse_rescaled_target_value, eps=1e-3))

                self.state_action_value = burn_in.select_state_value_action(
                    q_value=self.reformed_main_q_value, action=self.reformed_action, num_action=self.num_action)
                
                self.unmasked_td_error = (self.target_value - self.state_action_value) ** 2
                self.td_error = self.unmasked_td_error * tf.to_float(~self.reformed_mask)
                self.value_loss = tf.reduce_mean(self.td_error)

                
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.value_loss)    

        self.main_target = utils.main_to_target(f'{model_name}/main', f'{model_name}/target')
        self.global_to_session = utils.copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def target_to_main(self):
        self.sess.run(self.main_target)
            
    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_policy_and_action(self, state, h, c, previous_action, epsilon):

        state = np.stack(state) / 255
        one_q_value, one_h, one_c = self.sess.run(
            [self.one_q_value, self.one_h, self.one_c],
            feed_dict={
                self.s_ph: [state],
                self.pa_ph: [previous_action],
                self.initial_h_ph: [h],
                self.initial_c_ph: [c]})

        main_q_value = one_q_value[0]

        if np.random.rand() > epsilon:
            action = np.argmax(main_q_value, axis=0)
        else:
            action = np.random.choice(self.num_action)

        return action, main_q_value, main_q_value[action], one_h[0], one_c[0]

    def target_main_test(self, state, previous_action, initial_h, initial_c,
              action, reward, done):

        state = np.stack(state) / 255
        previous_action = np.stack(previous_action)
        initial_h = np.stack(initial_h)
        initial_c = np.stack(initial_c)
        action = np.stack(action)
        reward = np.stack(reward)
        done = np.stack(done)

        train_main_q_value, train_next_main_q_value, train_target_q_value = self.sess.run(
            [self.train_main_q_value, self.train_next_main_q_value, self.train_target_q_value],
            feed_dict={
                self.trajectory_main_s_ph: state,
                self.trajectory_main_pa_ph: previous_action,
                self.trajectory_main_initial_h_ph: initial_h,
                self.trajectory_main_initial_c_ph: initial_c,

                self.trajectory_target_s_ph: state,
                self.trajectory_target_pa_ph: previous_action,
                self.trajectory_target_initial_h_ph: initial_h[:, 0],
                self.trajectory_target_initial_c_ph: initial_c[:, 0],
                self.trajectory_target_done: done,

                self.trajectory_reward: reward,
                self.trajectory_action: action,
                self.trajectory_done: done})
        print('main')
        print(train_main_q_value)
        print('next main')
        print(train_next_main_q_value)
        print('target')
        print(train_target_q_value)
        print('-------------------')

    def train(self, state, previous_action, initial_h, initial_c,
              action, reward, done, mask):

        state = np.stack(state) / 255
        previous_action = np.stack(previous_action)
        initial_h = np.stack(initial_h)
        initial_c = np.stack(initial_c)
        action = np.stack(action)
        reward = np.stack(reward)
        done = np.stack(done)
        mask = np.stack(mask)

        loss, _ = self.sess.run(
            [self.value_loss, self.train_op],
            feed_dict={
                self.trajectory_main_s_ph: state,
                self.trajectory_main_pa_ph: previous_action,
                self.trajectory_main_initial_h_ph: initial_h,
                self.trajectory_main_initial_c_ph: initial_c,

                self.trajectory_target_s_ph: state,
                self.trajectory_target_pa_ph: previous_action,
                self.trajectory_target_initial_h_ph: initial_h[:, 0],
                self.trajectory_target_initial_c_ph: initial_c[:, 0],
                self.trajectory_target_done: done,
                
                self.trajectory_reward: reward,
                self.trajectory_done: done,
                self.trajectory_action: action,
                self.trajectory_mask: mask})

    def train_per(self, state, previous_action, initial_h, initial_c,
              action, reward, done, weight):

        print(np.stack(state).shape)

        # state = np.stack(state) / 255
        # previous_action = np.stack(previous_action)
        # initial_h = np.stack(initial_h)
        # initial_c = np.stack(initial_c)
        # action = np.stack(action)
        # reward = np.stack(reward)
        # done = np.stack(done)
        # weight = np.stack(weight)

        # loss, td_error, _ = self.sess.run(
        #     [self.value_loss, self.td_error, self.train_op],
        #     feed_dict={
        #         self.trajectory_main_s_ph: state,
        #         self.trajectory_main_pa_ph: previous_action,
        #         self.trajectory_main_initial_h_ph: initial_h,
        #         self.trajectory_main_initial_c_ph: initial_c,

        #         self.trajectory_target_s_ph: state,
        #         self.trajectory_target_pa_ph: previous_action,
        #         self.trajectory_target_initial_h_ph: initial_h[:, 0],
        #         self.trajectory_target_initial_c_ph: initial_c[:, 0],
        #         self.trajectory_target_done: done,

        #         self.trajectory_reward: reward,
        #         self.trajectory_action: action,
        #         self.trajectory_done: done,
        #         self.trajectory_weight: weight})

        # td_error = np.mean(td_error, axis=1)

        # return loss, td_error

    def get_td_error(self, state, previous_action, initial_h, initial_c,
                     action, reward, done):

        state = np.stack(state) / 255
        previous_action = np.stack(previous_action)
        initial_h = np.stack(initial_h)
        initial_c = np.stack(initial_c)
        action = np.stack(action)
        reward = np.stack(reward)
        done = np.stack(done)

        td_error = self.sess.run(
            self.td_error,
            feed_dict={
                self.trajectory_main_s_ph: [state],
                self.trajectory_main_pa_ph: [previous_action],
                self.trajectory_main_initial_h_ph: [initial_h],
                self.trajectory_main_initial_c_ph: [initial_c],

                self.trajectory_target_s_ph: [state],
                self.trajectory_target_pa_ph: [previous_action],
                self.trajectory_target_initial_h_ph: [initial_h[0]],
                self.trajectory_target_initial_c_ph: [initial_c[0]],
                self.trajectory_target_done: [done],

                self.trajectory_reward: [reward],
                self.trajectory_action: [action],
                self.trajectory_done: [done]})
        td_error = np.mean(td_error)
        td_error = np.abs(td_error)
        return td_error
