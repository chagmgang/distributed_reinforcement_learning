from model import impala_actor_critic
from optimizer import vtrace

import tensorflow as tf
import numpy as np

import utils

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

                self.s_ph = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
                self.pa_ph = tf.placeholder(tf.int32, shape=[None])
                self.initial_h_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_hidden_size])
                self.initial_c_ph = tf.placeholder(tf.float32, shape=[None, self.lstm_hidden_size])

                self.t_s_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, *self.input_shape])
                self.t_pa_ph = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.t_initial_h_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.lstm_hidden_size])
                self.t_initial_c_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.lstm_hidden_size])
                self.a_ph = tf.placeholder(tf.int32, shape=[None, self.trajectory])
                self.r_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory])
                self.d_ph = tf.placeholder(tf.bool, shape=[None, self.trajectory])
                self.b_ph = tf.placeholder(tf.float32, shape=[None, self.trajectory, self.num_action])

                if reward_clipping == 'abs_one':
                    self.clipped_r_ph = tf.clip_by_value(self.r_ph, -1.0, 1.0)
                elif reward_clipping == 'soft_asymmetric':
                    squeezed = tf.tanh(self.r_ph / 5.0)
                    self.clipped_r_ph = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.

                self.discounts = tf.to_float(~self.d_ph) * self.discount_factor

                self.policy, self.c, self.h, self.unrolled_first_policy, \
                    self.unrolled_first_value, self.unrolled_middle_policy,\
                        self.unrolled_middle_value, self.unrolled_last_policy,\
                            self.unrolled_last_value = impala_actor_critic.build_network(
                                                        state=self.s_ph, previous_action=self.pa_ph, trajectory=self.trajectory,
                                                        initial_h=self.initial_h_ph, initial_c=self.initial_c_ph,
                                                        num_action=self.num_action, lstm_hidden_size=self.lstm_hidden_size,
                                                        trajectory_state=self.t_s_ph, trajectory_previous_action=self.t_pa_ph,
                                                        trajectory_initial_h=self.t_initial_h_ph, trajectory_initial_c=self.t_initial_c_ph)

                self.unrolled_first_action, self.unrolled_middle_action, self.unrolled_last_action = vtrace.split_data(self.a_ph)
                self.unrolled_first_reward, self.unrolled_middle_reward, self.unrolled_last_reward = vtrace.split_data(self.clipped_r_ph)
                self.unrolled_first_discounts, self.unrolled_middle_discounts, self.unrolled_last_discounts = vtrace.split_data(self.discounts)
                self.unrolled_first_behavior_policy, self.unrolled_middle_behavior_policy, self.unrolled_last_behavior_policy = vtrace.split_data(self.b_ph)

                self.vs, self.clipped_rho = vtrace.from_softmax(
                                                behavior_policy_softmax=self.unrolled_first_behavior_policy, target_policy_softmax=self.unrolled_first_policy,
                                                actions=self.unrolled_first_action, discounts=self.unrolled_first_discounts, rewards=self.unrolled_first_reward,
                                                values=self.unrolled_first_value, next_values=self.unrolled_middle_value, action_size=self.num_action)

                self.vs_plus_1, _ = vtrace.from_softmax(
                                                behavior_policy_softmax=self.unrolled_middle_behavior_policy, target_policy_softmax=self.unrolled_middle_policy,
                                                actions=self.unrolled_middle_action, discounts=self.unrolled_middle_discounts, rewards=self.unrolled_middle_reward,
                                                values=self.unrolled_middle_value, next_values=self.unrolled_last_value, action_size=self.num_action)

                self.pg_advantage = tf.stop_gradient(
                    self.clipped_rho * \
                        (self.unrolled_first_reward + self.unrolled_first_discounts * self.vs_plus_1 - self.unrolled_first_value))
                
                self.pi_loss = vtrace.compute_policy_gradient_loss(
                    softmax=self.unrolled_first_policy,
                    actions=self.unrolled_first_action,
                    advantages=self.pg_advantage,
                    output_size=self.num_action)
                self.baseline_loss = vtrace.compute_baseline_loss(
                    vs=tf.stop_gradient(self.vs),
                    value=self.unrolled_first_value)
                self.entropy = vtrace.compute_entropy_loss(
                    softmax=self.unrolled_first_policy)

                self.total_loss = self.pi_loss + self.baseline_loss * self.baseline_loss_coef + self.entropy * self.entropy_coef

            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(self.start_learning_rate, self.num_env_frames, self.learning_frame, self.end_learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, momentum=0, epsilon=0.1)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.total_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variable), global_step=self.num_env_frames)

        self.global_to_session = utils.copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def save_weights(self, path):
        self.saver.save(self.sess, path)

    def load_weights(self, path):
        self.saver.restore(self.sess, path)

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_policy_and_action(self, state, previous_action, h, c):
        normalized_state = np.stack(state) / 255
        policy, result_c, result_h = self.sess.run(
            [self.policy, self.c, self.h], feed_dict={
                                            self.s_ph: [normalized_state],
                                            self.pa_ph: [previous_action],
                                            self.initial_h_ph: [h],
                                            self.initial_c_ph: [c]})
        policy = policy[0]
        result_c = result_c[0]
        result_h = result_h[0]
        action = np.random.choice(self.num_action, p=policy)
        return action, policy, max(policy), result_h, result_c

    def train(self, state, reward, action, done, behavior_policy, previous_action, initial_h, initial_c):
        normalized_state = np.stack(state) / 255
        feed_dict={
            self.t_s_ph: normalized_state,
            self.t_pa_ph: previous_action,
            self.t_initial_h_ph: initial_h,
            self.t_initial_c_ph: initial_c,
            self.a_ph: action,
            self.d_ph: done,
            self.r_ph: reward,
            self.b_ph: behavior_policy}

        pi_loss, value_loss, entropy, learning_rate, _ = self.sess.run(
            [self.pi_loss, self.baseline_loss, self.entropy, self.learning_rate, self.train_op],
            feed_dict=feed_dict)
        
        return pi_loss, value_loss, entropy, learning_rate