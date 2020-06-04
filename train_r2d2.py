import gym
import json
import time
import utils

import tensorflow as tf
import numpy as np

from distributed_queue import buffer_queue
from agent import r2d2
from tensorboardX import SummaryWriter

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id. Use -1 for local training")
flags.DEFINE_enum('job_name', 
                  'learner',
                  ['learner', 'actor'],
                  'Job name. Ignore when task is set to -1')
            
def main(_):
    data = json.load(open('config.json'))
    data = data['r2d2']
    utils.check_properties(data)

    local_job_device = f'/job:{FLAGS.job_name}/task:{FLAGS.task}'
    shared_job_device = '/job:learner/task:0'
    is_learner = FLAGS.job_name == 'learner'

    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:{}'.format(data['server_port']+1+i) for i in range(data['num_actors'])],
        'learner': ['{}:{}'.format(data['server_ip'], data['server_port'])]})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task)

    with tf.device(shared_job_device):

        with tf.device('/cpu'):
            r2d2_queue = buffer_queue.R2D2FIFOQueue(
                seq_len=data['seq_len'],
                input_shape=data['model_input'],
                output_size=data['model_output'],
                queue_size=data['queue_size'],
                batch_size=data['batch_size'],
                num_actors=data['num_actors'],
                lstm_size=data['lstm_size'])

        learner = r2d2.Agent(
            seq_len=data['seq_len'],
            burn_in=data['burn_in'],
            input_shape=data['model_input'],
            num_action=data['model_output'],
            lstm_size=data['lstm_size'],
            discount_factor=data['discount_factor'],
            start_learning_rate=data['start_learning_rate'],
            end_learning_rate=data['end_learning_rate'],
            learning_frame=data['learning_frame'],
            gradient_clip_norm=data['gradient_clip_norm'],
            model_name='learner',
            learner_name='learner')

    with tf.device(local_job_device):

        actor = r2d2.Agent(
            seq_len=data['seq_len'],
            burn_in=data['burn_in'],
            input_shape=data['model_input'],
            num_action=data['model_output'],
            lstm_size=data['lstm_size'],
            discount_factor=data['discount_factor'],
            start_learning_rate=data['start_learning_rate'],
            end_learning_rate=data['end_learning_rate'],
            learning_frame=data['learning_frame'],
            gradient_clip_norm=data['gradient_clip_norm'],
            model_name=f'actor_{FLAGS.task}',
            learner_name='learner')

    sess = tf.Session(server.target)
    r2d2_queue.set_session(sess)
    learner.set_session(sess)

    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        learner.main_to_target()
        writer = SummaryWriter('runs/learner')
        per_replay_buffer = buffer_queue.Memory(
            capacity=int(1e5))

        buffer_step = 0
        train_step = 0

        while True:

            print(buffer_step, train_step, r2d2_queue.get_size())

            if r2d2_queue.get_size() > data['batch_size']:
                queue_data = r2d2_queue.sample_batch()
                for i in range(data['batch_size']):
                    buffer_step += 1

                    td_error = learner.get_td_error(
                        state=queue_data.state[i],
                        previous_action=queue_data.previous_action[i],
                        action=queue_data.action[i],
                        reward=queue_data.reward[i],
                        done=queue_data.done[i],
                        h=queue_data.previous_h[i],
                        c=queue_data.previous_c[i])

                    per_replay_buffer.add(
                        td_error,
                        [queue_data.state[i], queue_data.previous_action[i],
                         queue_data.action[i], queue_data.reward[i],
                         queue_data.done[i], queue_data.previous_h[i],
                         queue_data.previous_c[i]])

            if buffer_step > data['batch_size'] * 2:

                s = time.time()

                train_step += 1

                minibatch, idxs, weight = per_replay_buffer.sample(data['batch_size'])
                
                batch_state = []
                batch_previous_action = []
                batch_action = []
                batch_reward = []
                batch_done = []
                batch_previous_h = []
                batch_previous_c = []

                for b in minibatch:
                    batch_state.append(b[0])
                    batch_previous_action.append(b[1])
                    batch_action.append(b[2])
                    batch_reward.append(b[3])
                    batch_done.append(b[4])
                    batch_previous_h.append(b[5])
                    batch_previous_c.append(b[6])

                loss, td_error = learner.train(
                    state=batch_state,
                    previous_action=batch_previous_action,
                    action=batch_action,
                    h=batch_previous_h,
                    c=batch_previous_c,
                    reward=batch_reward,
                    done=batch_done,
                    weight=weight)

                writer.add_scalar('data/loss', loss, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)

                if i in range(len(idxs)):
                    per_replay_buffer.update(
                        idxs[i], td_error[i])

                if train_step % 100 == 0:
                    learner.main_to_target()

    else:
        trajectory_buffer = buffer_queue.R2D2TrajectoryBuffer(
            seq_len=data['seq_len'])

        env = gym.make(data['env'][FLAGS.task])
        episode = 0
        epsilon = 1
        score = 0

        state = env.reset()
        state = (np.stack(state) * 255).astype(np.int32)
        state = [state[0], state[2]]
        previous_action = 0
        previous_h = np.zeros(data['lstm_size'])
        previous_c = np.zeros(data['lstm_size'])

        writer = SummaryWriter('runs/{}/actor_{}'.format(data['env'][FLAGS.task], FLAGS.task))

        while True:

            trajectory_buffer.init()
            actor.parameter_sync()

            for _ in range(data['seq_len']):

                action, q_value, h, c = actor.get_action(
                    state=state, h=previous_h, c=previous_c,
                    previous_action=previous_action, epsilon=epsilon)

                next_state, reward, done, _ = env.step(action)
                next_state = (np.stack(next_state) * 255).astype(np.int32)
                next_state = [next_state[0], next_state[2]]

                score += reward

                trajectory_buffer.append(
                    state=state,
                    previous_action=previous_action,
                    action=action,
                    reward=reward,
                    done=done,
                    initial_h=previous_h,
                    initial_c=previous_c)

                state = next_state
                previous_action = action
                previous_c = c
                previous_h = h

                if done:
                    print(episode, score)
                    writer.add_scalar('data/score', score, episode)
                    writer.add_scalar('data/epsilon', epsilon, episode)
                    episode += 1
                    score = 0
                    epsilon = 1.0 / (0.1 * episode + 1)
                    state = env.reset()
                    state = (np.stack(state) * 255).astype(np.int32)
                    state = [state[0], state[2]]
                    previous_action = 0
                    previous_h = np.zeros(data['lstm_size'])
                    previous_c = np.zeros(data['lstm_size'])
                    done = False

            trajectory_data = trajectory_buffer.extract()
            r2d2_queue.append_to_queue(
                task=FLAGS.task,
                unrolled_state=trajectory_data['state'],
                unrolled_previous_action=trajectory_data['previous_action'],
                unrolled_action=trajectory_data['action'],
                unrolled_reward=trajectory_data['reward'],
                unrolled_done=trajectory_data['done'],
                unrolled_previous_h=trajectory_data['initial_h'],
                unrolled_previous_c=trajectory_data['initial_c'])


if __name__ == '__main__':
    tf.app.run()