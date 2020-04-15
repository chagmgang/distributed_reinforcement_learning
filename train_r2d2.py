import json
import time
import utils
import wrappers

import tensorflow as tf
import numpy as np

from tensorboardX import SummaryWriter
from distributed_queue import buffer_queue
from agent import r2d2


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id. Use -1 for local training")
flags.DEFINE_enum('job_name', 
                  'learner',
                  ['learner', 'actor'],
                  'Job name. Ignore when task is set to -1')

def main(_):
    data = json.load(open('config.json'))
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

        learner = r2d2.Agent(
            trajectory=data['trajectory'],
            input_shape=data['model_input'],
            num_action=data['model_output'],
            lstm_hidden_size=data['lstm_size'],
            discount_factor=data['discount_factor'],
            start_learning_rate=data['start_learning_rate'],
            end_learning_rate=data['end_learning_rate'],
            learning_frame=data['learning_frame'],
            baseline_loss_coef=data['baseline_loss_coef'],
            entropy_coef=data['entropy_coef'],
            gradient_clip_norm=data['gradient_clip_norm'],
            reward_clipping=data['reward_clipping'],
            model_name='learner',
            learner_name='learner')

    if not is_learner:

        with tf.device(local_job_device):
            actor = r2d2.Agent(
                trajectory=data['trajectory'],
                input_shape=data['model_input'],
                num_action=data['model_output'],
                lstm_hidden_size=data['lstm_size'],
                discount_factor=data['discount_factor'],
                start_learning_rate=data['start_learning_rate'],
                end_learning_rate=data['end_learning_rate'],
                learning_frame=data['learning_frame'],
                baseline_loss_coef=data['baseline_loss_coef'],
                entropy_coef=data['entropy_coef'],
                gradient_clip_norm=data['gradient_clip_norm'],
                reward_clipping=data['reward_clipping'],
                model_name=f'actor_{FLAGS.task}',
                learner_name='learner')

    sess = tf.Session(server.target)
    queue.set_session(sess)
    learner.set_session(sess)

    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        learner.target_to_main()

        train_step = 0
        writer = SummaryWriter('runs/learner')
        while True:
            size = queue.get_size()
            if size > 3 * data['batch_size']:
                print('train {}'.format(train_step))
                train_step += 1
                batch = queue.sample_batch()
                s = time.time()
                loss, learning_rate = learner.train(
                    state=np.stack(batch.state),
                    reward=np.stack(batch.reward),
                    action=np.stack(batch.action),
                    done=np.stack(batch.done),
                    previous_action=np.stack(batch.previous_action),
                    initial_h=np.stack(batch.previous_h),
                    initial_c=np.stack(batch.previous_c))

                if train_step % 1000 == 0:
                    learner.target_to_main()

                writer.add_scalar('data/loss', loss, train_step)
                writer.add_scalar('data/learning_rate', learning_rate, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)

    else:

        trajectory = utils.UnrolledTrajectory()
        env = wrappers.make_uint8_env(data['env'][FLAGS.task])
        state = env.reset()
        previous_action = 0
        previous_h = np.zeros([data['lstm_size']])
        previous_c = np.zeros([data['lstm_size']])

        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        lives = 5

        epsilon = 1.0

        writer = SummaryWriter('runs/{}/actor_{}'.format(data['env'][FLAGS.task], FLAGS.task))

        while True:

            trajectory.initialize()
            actor.parameter_sync()

            for _ in range(data['trajectory']):

                action, q_value, max_q, h, c = actor.get_policy_and_action(
                    state=state, previous_action=previous_action,
                    h=previous_h, c=previous_c, epsilon=epsilon)

                episode_step += 1
                total_max_prob += max_q

                next_state, reward, done, info = env.step(action % data['available_action'][FLAGS.task])

                score += reward

                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                trajectory.append(
                    state=state, next_state=next_state, reward=r,
                    done=d, action=action, behavior_policy=q_value,
                    previous_action=previous_action,
                    initial_h=previous_h, initial_c=previous_c)

                state = next_state
                previous_action = action
                previous_h = h
                previous_c = c
                lives = info['ale.lives']

                if done:
                    print(episode, score)
                    writer.add_scalar('data/{}/prob'.format(data['env'][FLAGS.task]), total_max_prob / episode_step, episode)
                    writer.add_scalar('data/{}/score'.format(data['env'][FLAGS.task]), score, episode)
                    writer.add_scalar('data/{}/episode_step'.format(data['env'][FLAGS.task]), episode_step, episode)
                    writer.add_scalar('data/{}/epsilon'.format(data['env'][FLAGS.task]), epsilon, episode)

                    episode += 1
                    score = 0
                    episode_step = 0
                    total_max_prob = 0

                    state = env.reset()
                    previous_action = 0
                    previous_h = np.zeros([data['lstm_size']])
                    previous_c = np.zeros([data['lstm_size']])

            unrolled_data = trajectory.extract()
            queue.append_to_queue(
                task=FLAGS.task,
                unrolled_state=unrolled_data['state'],
                unrolled_next_state=unrolled_data['next_state'],
                unrolled_reward=unrolled_data['reward'],
                unrolled_done=unrolled_data['done'],
                unrolled_behavior_policy=unrolled_data['behavior_policy'],
                unrolled_action=unrolled_data['action'],
                unrolled_previous_action=unrolled_data['previous_action'],
                unrolled_previous_h=unrolled_data['initial_h'],
                unrolled_previous_c=unrolled_data['initial_c'])

if __name__ == '__main__':
    tf.app.run()