import json
import time
import utils
import wrappers

import tensorflow as tf
import numpy as np

from agent import a3c
from distributed_queue import buffer_queue
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
    data = data['a3c']
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
            queue = buffer_queue.A3CFIFOQueue(
                    trajectory_size=data['trajectory'],
                    input_shape=data['model_input'],
                    output_size=data['model_output'],
                    num_actors=data['num_actors'])

        learner = a3c.Agent(
                input_shape=data['model_input'],
                num_action=data['model_output'],
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

    with tf.device(local_job_device):

        actor = a3c.Agent(
                input_shape=data['model_input'],
                num_action=data['model_output'],
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
    learner.set_session(sess)
    queue.set_session(sess)
    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        writer = SummaryWriter('runs/learner')
        train_step = 0
        while True:

            if queue.get_size:
                train_step += 1
                batch = queue.sample_batch()
                pi_loss, value_loss, entropy, learning_rate = learner.train(
                        state=batch.state[0],
                        next_state=batch.next_state[0],
                        previous_action=batch.previous_action[0],
                        action=batch.action[0],
                        reward=batch.reward[0],
                        done=batch.done[0])

                writer.add_scalar('data/pi_loss', pi_loss, train_step)
                writer.add_scalar('data/value_loss', value_loss, train_step)
                writer.add_scalar('data/entropy', entropy, train_step)
                writer.add_scalar('data/lr', learning_rate, train_step)

                print('#########')
                print(f'pi loss    : {pi_loss}')
                print(f'value loss : {value_loss}')
                print(f'entropy    : {entropy}')
                print(f'lr         : {learning_rate}')
                print(f'step       : {train_step}')

    else:

        writer = SummaryWriter('runs/{}/actor_{}'.format(data['env'][FLAGS.task], FLAGS.task))

        trajectory = utils.UnrolledA3CTrajectory()
        env = wrappers.make_uint8_env(data['env'][FLAGS.task])
        state = env.reset()
        previous_action = 0

        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        lives = 5

        while True:

            trajectory.initialize()
            actor.parameter_sync()

            for _ in range(data['trajectory']):

                action, policy, max_prob = actor.get_policy_and_action(
                        state=state,
                        previous_action=previous_action)

                episode_step += 1
                total_max_prob += max_prob

                next_state, reward, done, info = env.step(
                        action % data['available_action'][FLAGS.task])

                score += reward

                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                trajectory.append(
                        state=state, next_state=next_state,
                        previous_action=previous_action, action=action,
                        reward=r, done=d)

                state = next_state
                previous_action = action
                lives = info['ale.lives']

                if done:
                    print(score, episode)
                    writer.add_scalar('data/{}/prob'.format(data['env'][FLAGS.task]), total_max_prob / episode_step, episode)
                    writer.add_scalar('data/{}/score'.format(data['env'][FLAGS.task]), score, episode)
                    writer.add_scalar('data/{}/episode_step'.format(data['env'][FLAGS.task]), episode_step, episode)
                    episode += 1
                    score = 0
                    episode_step = 0
                    total_max_prob = 0
                    state = env.reset()
                    previous_action = 0
                    lives = 5

            unrolled_data = trajectory.extract()
            queue.append_to_queue(
                    task=FLAGS.task,
                    unrolled_state=unrolled_data['state'],
                    unrolled_next_state=unrolled_data['next_state'],
                    unrolled_previous_action=unrolled_data['previous_action'],
                    unrolled_action=unrolled_data['action'],
                    unrolled_reward=unrolled_data['reward'],
                    unrolled_done=unrolled_data['done'])

if __name__ == '__main__':
    tf.app.run()
