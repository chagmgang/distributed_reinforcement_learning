from agent import apex

import tensorflow as tf
import numpy as np

import wrappers
import random
import utils
import json
import gym

from tensorboardX import SummaryWriter
from distributed_queue import buffer_queue

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id. Use -1 for local training")
flags.DEFINE_enum('job_name', 
                  'learner',
                  ['learner', 'actor'],
                  'Job name. Ignore when task is set to -1')

def main(_):
    data = json.load(open('config.json'))
    data = data['apex']
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
            queue = buffer_queue.FIFOQueue(
                trajectory=data['trajectory'],
                input_shape=data['model_input'],
                output_size=data['model_output'],
                queue_size=data['queue_size'],
                batch_size=data['batch_size'],
                num_actors=data['num_actors'],
                lstm_size=data['lstm_size'])

        learner = apex.Agent(
            input_shape=data['model_input'],
            num_action=data['model_output'],
            discount_factor=data['discount_factor'],
            gradient_clip_norm=data['gradient_clip_norm'],
            reward_clipping=data['reward_clipping'],
            start_learning_rate=data['start_learning_rate'],
            end_learning_rate=data['end_learning_rate'],
            learning_frame=data['learning_frame'],
            model_name='learner',
            learner_name='learner',
            memory_size=int(1e3))

    with tf.device(local_job_device):

        actor = apex.Agent(
            input_shape=data['model_input'],
            num_action=data['model_output'],
            discount_factor=data['discount_factor'],
            gradient_clip_norm=data['gradient_clip_norm'],
            reward_clipping=data['reward_clipping'],
            start_learning_rate=data['start_learning_rate'],
            end_learning_rate=data['end_learning_rate'],
            learning_frame=data['learning_frame'],
            model_name=f'actor_{FLAGS.task}',
            learner_name='learner',
            memory_size=int(1e3))

    sess = tf.Session(server.target)
    queue.set_session(sess)
    learner.set_session(sess)

    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        learner.target_to_main()
        train_step = 0
        buffer_size = 0

        writer = SummaryWriter('runs/learner')

        while True:
            size = queue.get_size()
            if size > 2:
                batch = queue.sample_batch()
                
                state = np.reshape(batch.state, (-1, *data['model_input']))
                next_state = np.reshape(batch.next_state, (-1, *data['model_input']))
                previous_action = np.reshape(batch.previous_action, (-1))
                action = np.reshape(batch.action, (-1))
                reward = np.reshape(batch.reward, (-1))
                done = np.reshape(batch.done, (-1))

                for s, ns, pa, a, r, d in zip(state, next_state, previous_action, action, reward, done):
                    buffer_size += 1
                    learner.append_to_memory(
                        state=s,
                        next_state=ns,
                        previous_action=pa,
                        action=a,
                        reward=r,
                        done=d)

            if buffer_size > 1000:
                train_step += 1
                loss, learning_rate = learner.train(data['batch_size'])
                writer.add_scalar('data/loss', loss, train_step)
                if train_step % 100:
                    learner.target_to_main()

    else:

        env = wrappers.make_uint8_env(data['env'][FLAGS.task])

        state = env.reset()
        previous_action = 0

        epsilon = 1.0
        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        lives = 5

        writer = SummaryWriter('runs/{}/actor_{}'.format(data['env'][FLAGS.task], FLAGS.task))

        while True:

            actor.parameter_sync()

            for _ in range(data['trajectory']):

                action, q_value, selected_q_value = actor.get_policy_and_action(
                    state=state, previous_action=previous_action, epsilon=epsilon)

                episode_step += 1
                total_max_prob += selected_q_value

                next_state, reward, done, info = env.step(action % data['available_action'][FLAGS.task])

                score += reward

                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                actor.append_to_memory(
                    state=state,
                    next_state=next_state,
                    previous_action=previous_action,
                    action=action,
                    reward=r,
                    done=d)

                state = next_state
                previous_action = action

                if done:
                    print(episode, score)
                    writer.add_scalar('data/{}/prob'.format(data['env'][FLAGS.task]), total_max_prob / episode_step, episode)
                    writer.add_scalar('data/{}/score'.format(data['env'][FLAGS.task]), score, episode)
                    writer.add_scalar('data/{}/episode_step'.format(data['env'][FLAGS.task]), episode_step, episode)
                    writer.add_scalar('data/{}/epsilon'.format(data['env'][FLAGS.task]), epsilon, episode)
                    episode_step = 0
                    total_max_prob = 0
                    episode += 1
                    score = 0
                    epsilon = 1 / (0.05 * episode + 1)
                    state = env.reset()
                    previous_action = 0

            sampled_data = actor.sample(data['trajectory'])
            queue.append_to_queue(
                task=FLAGS.task,
                unrolled_state=sampled_data['state'],
                unrolled_next_state=sampled_data['next_state'],
                unrolled_reward=sampled_data['reward'],
                unrolled_done=sampled_data['done'],
                unrolled_behavior_policy=np.zeros([data['trajectory'], data['model_output']]),
                unrolled_action=sampled_data['action'],
                unrolled_previous_action=sampled_data['previous_action'],
                unrolled_previous_h=np.zeros([data['trajectory'], data['lstm_size']]),
                unrolled_previous_c=np.zeros([data['trajectory'], data['lstm_size']]))


    

if __name__ == '__main__':
    tf.app.run()
