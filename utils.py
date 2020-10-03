import tensorflow as tf
import numpy as np

import collections

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

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def main_to_target(src, dst):
    src = get_vars(src)
    dst = get_vars(dst)
    main_target = tf.group([tf.assign(v_targ, v_main)
            for v_main, v_targ in zip(src, dst)])
    return main_target

def check_properties(data):
    ## check available_action < model_output
    for a in data['available_action']:
        assert data['model_output'] >= a

    ## check available_action size == actor size
    assert data['num_actors'] == len(data['available_action'])
    ## check actor size == env size
    assert data['num_actors'] == len(data['env'])
    ## check available_action size == env size
    assert len(data['available_action']) == len(data['env'])
    assert data['reward_clipping'] in ['abs_one', 'soft_asymmetric']

class UnrolledA3CTrajectory:

    def __init__(self):
        self.trajectory_data = collections.namedtuple(
                'trajectory_data',
                ['state', 'next_state', 'reward', 'done',
                 'action', 'previous_action'])

    def initialize(self):
        self.unroll_data = self.trajectory_data(
                [],[],[],[],[],[])

    def append(self, state, next_state, previous_action,
               action, reward, done):

        self.unroll_data.state.append(state)
        self.unroll_data.next_state.append(next_state)
        self.unroll_data.previous_action.append(previous_action)
        self.unroll_data.action.append(action)
        self.unroll_data.reward.append(reward)
        self.unroll_data.done.append(done)

    def extract(self):
        data = {
                'state': np.stack(self.unroll_data.state),
                'next_state': np.stack(self.unroll_data.next_state),
                'previous_action': np.stack(self.unroll_data.previous_action),
                'action': np.stack(self.unroll_data.action),
                'reward': np.stack(self.unroll_data.reward),
                'done': np.stack(self.unroll_data.done)}
        return data


class UnrolledTrajectory:

    def __init__(self):
        self.trajectory_data = collections.namedtuple(
                'trajectory_data',
                ['state', 'next_state', 'reward', 'done',
                 'action', 'behavior_policy', 'previous_action',
                 'initial_h', 'initial_c'])

    def initialize(self):
        self.unroll_data = self.trajectory_data(
            [], [], [], [],
                [], [], [] ,[], [])

    def append(self, state, next_state, reward, done,
               action, behavior_policy, previous_action, initial_h, initial_c):

        self.unroll_data.state.append(state)
        self.unroll_data.next_state.append(next_state)
        self.unroll_data.reward.append(reward)
        self.unroll_data.done.append(done)
        self.unroll_data.action.append(action)
        self.unroll_data.behavior_policy.append(behavior_policy)
        self.unroll_data.previous_action.append(previous_action)
        self.unroll_data.initial_h.append(initial_h)
        self.unroll_data.initial_c.append(initial_c)

    def extract(self):
        data = {
            'state': self.unroll_data.state,
            'next_state': self.unroll_data.next_state,
            'reward': self.unroll_data.reward,
            'done': self.unroll_data.done,
            'action': self.unroll_data.action,
            'behavior_policy': self.unroll_data.behavior_policy,
            'previous_action': self.unroll_data.previous_action,
            'initial_h': self.unroll_data.initial_h,
            'initial_c': self.unroll_data.initial_c}

        return data
