import tensorflow as tf
import numpy as np

import random
import collections

class R2D2FIFOQueue:

    def __init__(self, seq_len, input_shape, output_size,
                 queue_size, batch_size, num_actors, lstm_size):

        self.seq_len = seq_len
        self.input_shape = input_shape
        self.output_size = output_size
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.num_actors = num_actors
        self.lstm_size = lstm_size

        self.unrolled_state = tf.placeholder(tf.int32, [self.seq_len, *self.input_shape])
        self.unrolled_previous_action = tf.placeholder(tf.int32, [self.seq_len])
        self.unrolled_action = tf.placeholder(tf.int32, [self.seq_len])
        self.unrolled_reward = tf.placeholder(tf.float32, shape=[self.seq_len])
        self.unrolled_done = tf.placeholder(tf.bool, shape=[self.seq_len])
        self.unrolled_previous_h = tf.placeholder(tf.float32, shape=[self.seq_len, self.lstm_size])
        self.unrolled_previous_c = tf.placeholder(tf.float32, shape=[self.seq_len, self.lstm_size])

        self.queue = tf.FIFOQueue(
            queue_size,
            [self.unrolled_state.dtype,
             self.unrolled_previous_action.dtype,
             self.unrolled_action.dtype,
             self.unrolled_reward.dtype,
             self.unrolled_done.dtype,
             self.unrolled_previous_h.dtype,
             self.unrolled_previous_c.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()

        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                self.queue.enqueue(
                    [self.unrolled_state,
                     self.unrolled_previous_action,
                     self.unrolled_action,
                     self.unrolled_reward,
                     self.unrolled_done,
                     self.unrolled_previous_h,
                     self.unrolled_previous_c]))

        self.dequeue = self.queue.dequeue()

    def sample_batch(self):
        batch_tuple = collections.namedtuple('batch_tuple',
        ['state', 'previous_action', 'action', 'reward',
         'done', 'previous_h', 'previous_c'])

        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        unrolled_data = batch_tuple(
            [i[0] for i in batch],
            [i[1] for i in batch],
            [i[2] for i in batch],
            [i[3] for i in batch],
            [i[4] for i in batch],
            [i[5] for i in batch],
            [i[6] for i in batch])

        return unrolled_data

    def get_size(self):
        size = self.sess.run(self.queue_size)
        return size

    def set_session(self, sess):
        self.sess = sess

    def append_to_queue(self, task, unrolled_state, unrolled_previous_action,
                        unrolled_action, unrolled_reward, unrolled_done,
                        unrolled_previous_h, unrolled_previous_c):
        self.sess.run(
            self.enqueue_ops[task],
            feed_dict={
                self.unrolled_state: unrolled_state,
                self.unrolled_previous_action: unrolled_previous_action,
                self.unrolled_action: unrolled_action,
                self.unrolled_reward: unrolled_reward,
                self.unrolled_done: unrolled_done,
                self.unrolled_previous_h: unrolled_previous_h,
                self.unrolled_previous_c: unrolled_previous_c})
        

class R2D2TrajectoryBuffer:

    def __init__(self, seq_len):
        self.seq_len = seq_len
        
        self.state = collections.deque(maxlen=int(self.seq_len))
        self.previous_action = collections.deque(maxlen=int(self.seq_len))
        self.action = collections.deque(maxlen=int(self.seq_len))
        self.reward = collections.deque(maxlen=int(self.seq_len))
        self.done = collections.deque(maxlen=int(self.seq_len))
        self.initial_h = collections.deque(maxlen=int(self.seq_len))
        self.initial_c = collections.deque(maxlen=int(self.seq_len))

    def append(self, state, previous_action, action, reward, done, initial_h, initial_c):
        self.state.append(state)
        self.previous_action.append(previous_action)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.initial_h.append(initial_h)
        self.initial_c.append(initial_c)

    def init(self):
        self.state = collections.deque(maxlen=int(self.seq_len))
        self.previous_action = collections.deque(maxlen=int(self.seq_len))
        self.action = collections.deque(maxlen=int(self.seq_len))
        self.reward = collections.deque(maxlen=int(self.seq_len))
        self.done = collections.deque(maxlen=int(self.seq_len))
        self.initial_h = collections.deque(maxlen=int(self.seq_len))
        self.initial_c = collections.deque(maxlen=int(self.seq_len))

    def extract(self):
        data = {
            'state':self.state,
            'previous_action':self.previous_action,
            'action':self.action,
            'reward':self.reward,
            'done':self.done,
            'initial_h':self.initial_h,
            'initial_c':self.initial_c}
        return data

class ApexFIFOQueue:

    def __init__(self, trajectory, input_shape, output_size,
                 queue_size, batch_size, num_actors):

        self.trajectory = trajectory
        self.input_shape = input_shape
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.unrolled_state = tf.placeholder(tf.int32, shape=[self.trajectory, *self.input_shape])
        self.unrolled_next_state = tf.placeholder(tf.int32, shape=[self.trajectory, *self.input_shape])
        self.unrolled_previous_action = tf.placeholder(tf.int32, shape=[self.trajectory])
        self.unrolled_action = tf.placeholder(tf.int32, shape=[self.trajectory])
        self.unrolled_reward = tf.placeholder(tf.float32, shape=[self.trajectory])
        self.unrolled_done = tf.placeholder(tf.bool, shape=[self.trajectory])

        self.queue = tf.FIFOQueue(
            queue_size,
            [self.unrolled_state.dtype,
             self.unrolled_next_state.dtype,
             self.unrolled_previous_action.dtype,
             self.unrolled_action.dtype,
             self.unrolled_reward.dtype,
             self.unrolled_done.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()

        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                self.queue.enqueue(
                    [self.unrolled_state,
                     self.unrolled_next_state,
                     self.unrolled_previous_action,
                     self.unrolled_action,
                     self.unrolled_reward,
                     self.unrolled_done]))

        self.dequeue = self.queue.dequeue()

    def append_to_queue(self, task, unrolled_state, unrolled_next_state,
                        unrolled_previous_action, unrolled_action,
                        unrolled_reward, unrolled_done):
        self.sess.run(
            self.enqueue_ops[task],
            feed_dict={
                self.unrolled_state: unrolled_state,
                self.unrolled_next_state: unrolled_next_state,
                self.unrolled_previous_action: unrolled_previous_action,
                self.unrolled_action: unrolled_action,
                self.unrolled_reward: unrolled_reward,
                self.unrolled_done: unrolled_done})

    def sample_batch(self, batch_size):
        batch_tuple = collections.namedtuple(
            'batch_tuple',
            ['state', 'next_state', 'previous_action',
             'action', 'reward', 'done'])
        batch = [self.sess.run(self.dequeue) for i in range(batch_size)]
        unrolled_data = batch_tuple(
            [i[0] for i in batch],
            [i[1] for i in batch],
            [i[2] for i in batch],
            [i[3] for i in batch],
            [i[4] for i in batch],
            [i[5] for i in batch])

        return unrolled_data

    def get_size(self):
        size = self.sess.run(self.queue_size)
        return size

    def set_session(self, sess):
        self.sess = sess

class LocalBuffer:

    def __init__(self, capacity):
        self.state = collections.deque(maxlen=int(capacity))
        self.next_state = collections.deque(maxlen=int(capacity))
        self.previous_action = collections.deque(maxlen=int(capacity))
        self.action = collections.deque(maxlen=int(capacity))
        self.reward = collections.deque(maxlen=int(capacity))
        self.done = collections.deque(maxlen=int(capacity))

    def append(self, state, next_state, previous_action, action, reward, done):
        self.state.append(state)
        self.next_state.append(next_state)
        self.previous_action.append(previous_action)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

    def sample(self, batch_size):
        arange_list = np.arange(len(self.state))
        np.random.shuffle(arange_list)
        idxs = arange_list[:batch_size]

        state = [self.state[idx] for idx in idxs]
        next_state = [self.next_state[idx] for idx in idxs]
        previous_action = [self.previous_action[idx] for idx in idxs]
        action = [self.action[idx] for idx in idxs]
        reward = [self.reward[idx] for idx in idxs]
        done = [self.done[idx] for idx in idxs]

        data = {
            'state': state,
            'next_state': next_state,
            'previous_action': previous_action,
            'action': action,
            'reward': reward,
            'done': done}

        return data

    def __len__(self):
        return len(self.state)

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Memory(object):
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a, b = segment * i, segment * (i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class FIFOQueue:
    def __init__(self, trajectory, input_shape, output_size,
                 queue_size, batch_size, num_actors, lstm_size):
        
        self.trajectory = trajectory
        self.input_shape = input_shape
        self.output_size = output_size
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        
        self.unrolled_state = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape])
        self.unrolled_next_state = tf.placeholder(tf.uint8, [self.trajectory, *self.input_shape])
        self.unrolled_reward = tf.placeholder(tf.float32, [self.trajectory])
        self.unrolled_done = tf.placeholder(tf.bool, [self.trajectory])
        self.unrolled_behavior_policy = tf.placeholder(tf.float32, [self.trajectory, self.output_size])
        self.unrolled_action = tf.placeholder(tf.int32, [self.trajectory])
        self.unrolled_previous_action = tf.placeholder(tf.int32, [self.trajectory])
        self.unrolled_previous_h = tf.placeholder(tf.float32, shape=[self.trajectory, self.lstm_size])
        self.unrolled_previous_c = tf.placeholder(tf.float32, shape=[self.trajectory, self.lstm_size])

        self.queue = tf.FIFOQueue(
            queue_size,
            [self.unrolled_state.dtype,
            self.unrolled_next_state.dtype,
            self.unrolled_reward.dtype,
            self.unrolled_done.dtype,
            self.unrolled_behavior_policy.dtype,
            self.unrolled_action.dtype,
            self.unrolled_previous_action.dtype,
            self.unrolled_previous_h.dtype,
            self.unrolled_previous_c.dtype], shared_name='buffer')

        self.queue_size = self.queue.size()
        
        self.enqueue_ops = []
        for i in range(num_actors):
            self.enqueue_ops.append(
                self.queue.enqueue(
                    [self.unrolled_state,
                     self.unrolled_next_state,
                     self.unrolled_reward,
                     self.unrolled_done,
                     self.unrolled_behavior_policy,
                     self.unrolled_action,
                     self.unrolled_previous_action,
                     self.unrolled_previous_h,
                     self.unrolled_previous_c]))

        self.dequeue = self.queue.dequeue()

    def append_to_queue(self, task, unrolled_state, unrolled_next_state,
                        unrolled_reward, unrolled_done, unrolled_behavior_policy,
                        unrolled_action, unrolled_previous_action,
                        unrolled_previous_h, unrolled_previous_c):

        self.sess.run(
            self.enqueue_ops[task],
            feed_dict={
                self.unrolled_state: unrolled_state,
                self.unrolled_next_state: unrolled_next_state,
                self.unrolled_reward: unrolled_reward,
                self.unrolled_done: unrolled_done,
                self.unrolled_behavior_policy: unrolled_behavior_policy,
                self.unrolled_action: unrolled_action,
                self.unrolled_previous_action: unrolled_previous_action,
                self.unrolled_previous_h: unrolled_previous_h,
                self.unrolled_previous_c: unrolled_previous_c})

    def sample_batch(self):
        batch_tuple = collections.namedtuple('batch_tuple',
        ['state', 'next_state', 'reward', 'done',
         'behavior_policy', 'action', 'previous_action',
         'previous_h', 'previous_c'])

        batch = [self.sess.run(self.dequeue) for i in range(self.batch_size)]

        unroll_data = batch_tuple(
            [i[0] for i in batch],
            [i[1] for i in batch],
            [i[2] for i in batch],
            [i[3] for i in batch],
            [i[4] for i in batch],
            [i[5] for i in batch],
            [i[6] for i in batch],
            [i[7] for i in batch],
            [i[8] for i in batch])

        return unroll_data

    def get_size(self):
        size = self.sess.run(self.queue_size)
        return size

    def set_session(self, sess):
        self.sess = sess

if __name__ == '__main__':
    
    queue = FIFOQueue(
        20, [84, 84, 4], 3, 128, 32, 4)

    print(queue.unrolled_state)
