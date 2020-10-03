from model import actor_critic
from optimizer import a2c

import tensorflow as tf
import numpy as np

import utils

class Agent:

    def __init__(self, input_shape, num_action, discount_factor,
                 start_learning_rate, end_learning_rate,
                 learning_frame, baseline_loss_coef, entropy_coef,
                 gradient_clip_norm, reward_clipping, model_name, learner_name):

        self.input_shape = input_shape
        self.num_action = num_action
        self.discount_factor = discount_factor
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_frame = learning_frame
        self.baseline_loss_coef = baseline_loss_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip_norm = gradient_clip_norm

        with tf.variable_scope(model_name):

            with tf.device('cpu'):

                self.s_ph = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
                self.ns_ph = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
                self.pa_ph = tf.placeholder(tf.int32, shape=[None])
                self.npa_ph = tf.placeholder(tf.int32, shape=[None])
                
                self.r_ph = tf.placeholder(tf.float32, shape=[None])
                self.d_ph = tf.placeholder(tf.bool, shape=[None])
                self.a_ph = tf.placeholder(tf.int32, shape=[None])

                if reward_clipping == 'abs_one':
                    self.clipped_r_ph = tf.clip_by_value(self.r_ph, -1.0, 1.0)
                elif reward_clipping == 'soft_asymmetric':
                    squeezed = tf.tanh(self.r_ph / 5.0)
                    self.clipped_r_ph = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.

                self.discounts = tf.to_float(~self.d_ph) * self.discount_factor

                self.policy, self.value, _, self.next_value = actor_critic.build_network(
                        state=self.s_ph, next_state=self.ns_ph,
                        previous_action=self.pa_ph, next_previous_action=self.npa_ph,
                        num_action=self.num_action)

                self.pi_loss = a2c.compute_policy_loss(
                        policy=self.policy,
                        action=self.a_ph,
                        value=self.value,
                        next_value=self.next_value,
                        discounts=self.discounts,
                        reward=self.clipped_r_ph,
                        num_action=self.num_action)

                self.baseline_loss = a2c.compute_baseline_loss(
                        value=self.value,
                        next_value=self.next_value,
                        discounts=self.discounts,
                        reward=self.clipped_r_ph)

                self.entropy = a2c.compute_entropy_loss(
                        policy=self.policy)

                self.total_loss = self.pi_loss + self.baseline_loss * self.baseline_loss_coef + self.entropy * self.entropy_coef
                
            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(self.start_learning_rate, self.num_env_frames, self.learning_frame, self.end_learning_rate)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.total_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variable), global_step=self.num_env_frames)

        self.global_to_session = utils.copy_src_to_dst(learner_name, model_name)

    def train(self, state, next_state, previous_action, action, reward, done):
        normalized_state = np.stack(state) / 255
        normalized_next_state = np.stack(next_state) / 255

        pi_loss, value_loss, entropy, learning_rate, _ = self.sess.run(
                [self.pi_loss, self.baseline_loss, self.entropy,
                 self.learning_rate, self.train_op],
                feed_dict={
                    self.s_ph: normalized_state,
                    self.ns_ph: normalized_next_state,
                    self.pa_ph: previous_action,
                    self.npa_ph: action,
                    self.a_ph: action,
                    self.r_ph: reward,
                    self.d_ph: done})

        return pi_loss, value_loss, entropy, learning_rate

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_policy_and_action(self, state, previous_action):
        normalized_state = np.stack(state) / 255
        policy = self.sess.run(
                self.policy,
                feed_dict={
                    self.s_ph: [normalized_state],
                    self.pa_ph: [previous_action]})
        policy = policy[0]
        action = np.random.choice(self.num_action, p=policy)
        max_prob = policy[action]
        return action, policy, max_prob

    def parameter_sync(self):
        self.sess.run(self.global_to_session)
