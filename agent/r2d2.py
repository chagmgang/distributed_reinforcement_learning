from model import r2d2_lstm
from optimizer import burn_in as r2d2_optimizer
from distributed_queue import buffer_queue

import tensorflow as tf
import numpy as np

import gym
import utils

class Agent:

    def __init__(self, seq_len, burn_in, input_shape, num_action, lstm_size):
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.input_shape = input_shape
        self.num_action = num_action
        self.lstm_size = lstm_size

        self.s_ph = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
        self.h_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_size])
        self.c_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_size])
        self.pa_ph = tf.placeholder(tf.int32, shape=[None])

        self.main_s_ph = tf.placeholder(tf.float32, shape=[None, self.seq_len, *self.input_shape])
        self.main_h_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_size])
        self.main_c_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_size])
        self.main_d_ph = tf.placeholder(tf.bool, shape=[None, self.seq_len])
        self.main_pa_ph = tf.placeholder(tf.int32, shape=[None, self.seq_len])

        self.target_s_ph = tf.placeholder(tf.float32, shape=[None, self.seq_len, *self.input_shape])
        self.target_h_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_size])
        self.target_c_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_size])
        self.target_d_ph = tf.placeholder(tf.bool, shape=[None, self.seq_len])
        self.target_pa_ph = tf.placeholder(tf.int32, shape=[None, self.seq_len])

        self.reward_ph = tf.placeholder(tf.float32, shape=[None, self.seq_len])
        self.done_ph = tf.placeholder(tf.bool, shape=[None, self.seq_len])
        self.action_ph = tf.placeholder(tf.int32, shape=[None, self.seq_len])
        self.weight_ph = tf.placeholder(tf.float32, shape=[None])

        self.q_value, self.h, self.c, self.main_q, self.target_q = r2d2_lstm.build_network(
                s_ph=self.s_ph, h_ph=self.h_ph, c_ph=self.c_ph,
                pa_ph=self.pa_ph,
                main_s_ph=self.main_s_ph, main_h_ph=self.main_h_ph,
                main_c_ph=self.main_c_ph, main_d_ph=self.main_d_ph,
                main_pa_ph=self.main_pa_ph,
                target_s_ph=self.target_s_ph, target_h_ph=self.target_h_ph,
                target_c_ph=self.target_c_ph, target_d_ph=self.target_d_ph,
                target_pa_ph=self.target_pa_ph,
                lstm_size=self.lstm_size, num_action=self.num_action)

        self.discounts = tf.to_float(~self.done_ph) * 0.997

        burned_main_q = self.main_q[:, self.burn_in:]
        burned_target_q = self.target_q[:, self.burn_in:]
        burned_reward = self.reward_ph[:, self.burn_in:]
        burned_discounts = self.discounts[:, self.burn_in:]
        burned_action = self.action_ph[:, self.burn_in:]

        state_main_q = burned_main_q[:, :-1]
        next_state_main_q = burned_main_q[:, 1:]
        next_state_target_q = burned_target_q[:, 1:]
        action = burned_action[:, :-1]
        next_action = tf.argmax(next_state_main_q, axis=2)
        reward = burned_reward[:, :-1]
        discounts = burned_discounts[:, :-1]

        onehot_action = tf.one_hot(action, self.num_action)
        onehot_next_action = tf.one_hot(next_action, self.num_action)

        self.state_action_value = tf.reduce_sum(state_main_q * onehot_action, axis=2)
        self.next_state_action_value = tf.reduce_sum(next_state_target_q * onehot_next_action, axis=2)
        self.rescaled_next_state_action_value = r2d2_optimizer.inverse_value_function_rescaling(
                x=self.next_state_action_value, eps=1e-3)
        self.rescaled_target_value = tf.stop_gradient(self.rescaled_next_state_action_value * discounts + reward)
        self.target_value = r2d2_optimizer.value_function_rescaling(
                x=self.rescaled_target_value, eps=1e-3)
        self.unweighted_loss = tf.reduce_mean(((self.target_value - self.state_action_value) ** 2), axis=1)
        self.value_loss = tf.reduce_mean(self.unweighted_loss * self.weight_ph)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.train_op = self.optimizer.minimize(self.value_loss)

        self.main_target = utils.main_to_target('main', 'target')

    def get_td_error(self, state, previous_action, action, h, c, reward, done):
        state = np.stack([state]) / 255
        previous_action = np.stack([previous_action])
        action = np.stack([action])
        h = np.stack([h])
        c = np.stack([c])
        reward = np.stack([reward])
        done = np.stack([done])

        target_value, state_action_value = self.sess.run(
            [self.target_value, self.state_action_value],
            feed_dict={
                self.main_s_ph: state,
                self.main_h_ph: np.stack(h)[:, 0],
                self.main_c_ph: np.stack(c)[:, 0],
                self.main_d_ph: done,
                self.main_pa_ph: previous_action,

                self.target_s_ph: state,
                self.target_h_ph: np.stack(h)[:, 0],
                self.target_c_ph: np.stack(c)[:, 0],
                self.target_d_ph: done,
                self.target_pa_ph: previous_action,

                self.reward_ph: reward,
                self.done_ph: done,
                self.action_ph: action})

        td_error = np.mean(target_value - state_action_value)
        td_error = np.abs(td_error)
        return td_error

    def train(self, state, previous_action, action, h, c, reward, done, weight):
        state = np.stack(state) / 255
        loss, target_value, state_action_value, _ = self.sess.run(
            [self.value_loss, self.target_value, self.state_action_value, self.train_op],
            feed_dict={
                self.main_s_ph: state,
                self.main_h_ph: np.stack(h)[:, 0],
                self.main_c_ph: np.stack(c)[:, 0],
                self.main_d_ph: done,
                self.main_pa_ph: previous_action,

                self.target_s_ph: state,
                self.target_h_ph: np.stack(h)[:, 0],
                self.target_c_ph: np.stack(c)[:, 0],
                self.target_d_ph: done,
                self.target_pa_ph: previous_action,

                self.reward_ph: reward,
                self.done_ph: done,
                self.action_ph: action,
                self.weight_ph: weight})

        td_error = np.mean(target_value - state_action_value, axis=1)
        td_error = np.abs(td_error)

        return loss, td_error

    def main_to_target(self):
        self.sess.run(self.main_target)

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state, h, c, previous_action, epsilon):
        state = np.stack(state) / 255
        h = np.stack(h)
        c = np.stack(c)

        q_value, h, c = self.sess.run(
            [self.q_value, self.h, self.c],
            feed_dict={
                self.s_ph: [state],
                self.h_ph: [h],
                self.c_ph: [c],
                self.pa_ph: [previous_action]})

        q_value = q_value[0]

        if np.random.rand() > epsilon:
            action = np.argmax(q_value)
        else:
            action = np.random.choice(self.num_action)

        return action, q_value, h[0], c[0]

    def main_q_value_test(self, state, h, c, done, previous_action):
        state = np.stack(state) / 255

        main_q = self.sess.run(
            self.main_q,
            feed_dict={
                self.main_s_ph: [state],
                self.main_h_ph: [h[0]],
                self.main_c_ph: [c[0]],
                self.main_d_ph: [done],
                self.main_pa_ph: [previous_action]})
        return main_q[0]

    def target_q_value_test(self, state, h, c, done, previous_action):
        state = np.stack(state) / 255
        target_q = self.sess.run(
            self.target_q,
            feed_dict={
                self.target_s_ph: [state],
                self.target_h_ph: [h[0]],
                self.target_c_ph: [c[0]],
                self.target_d_ph: [done],
                self.target_pa_ph: [previous_action]})
        return target_q[0]
