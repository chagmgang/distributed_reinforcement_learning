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
    print(data)
    
    local_job_device = f'/job:{FLAGS.job_name}/task:{FLAGS.task}'
    shared_job_device = '/job:learner/task:0'
    is_learner = FLAGS.job_name == 'learner'

    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:{}'.format(data['server_port']+1+i) for i in range(data['num_actors'])],
        'learner': ['{}:{}'.format(data['server_ip'], data['server_port'])]})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task)

    with tf.device(shared_job_device):
        with tf.device('/cpu'):
            apex_queue = buffer_queue.ApexFIFOQueue(
                trajectory=data['trajectory'],
                input_shape=data['model_input'],
                output_size=data['model_output'],
                queue_size=data['queue_size'],
                batch_size=data['batch_size'],
                num_actors=data['num_actors'])

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
            learner_name='learner')

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
            learner_name='learner')

    sess = tf.Session(server.target)
    apex_queue.set_session(sess)
    learner.set_session(sess)

    if not is_learner:
        actor.set_session(sess)

    if is_learner:
        import time

        learner.target_to_main()
        replay_buffer = buffer_queue.Memory(capacity=int(1e5))
        buffer_step = 0
        train_step = 0

        writer = SummaryWriter('runs/learner')

        while True:
            print(f'train step : {train_step} | buffer step : {buffer_step}')
            if apex_queue.get_size():

                buffer_step += 1

                from_actor = apex_queue.sample_batch(1)
                from_actor_state = from_actor.state[0]
                from_actor_next_state = from_actor.next_state[0]
                from_actor_previous_action = from_actor.previous_action[0]
                from_actor_action = from_actor.action[0]
                from_actor_reward = from_actor.reward[0]
                from_actor_done = from_actor.done[0]

                td_error = learner.get_td_error(
                    state=from_actor_state,
                    next_state=from_actor_next_state,
                    previous_action=from_actor_previous_action,
                    action=from_actor_action,
                    reward=from_actor_reward,
                    done=from_actor_done)

                for i in range(len(td_error)):
                    replay_buffer.add(
                        td_error[i],
                        [from_actor_state[i],
                         from_actor_next_state[i],
                         from_actor_previous_action[i],
                         from_actor_action[i],
                         from_actor_reward[i],
                         from_actor_done[i]])

            if buffer_step > 10:
                train_step += 1

                s = time.time()

                minibatch, idxs, is_weight = replay_buffer.sample(data['batch_size'])
                minibatch = np.array(minibatch)

                state = np.stack(minibatch[:, 0])
                next_state = np.stack(minibatch[:, 1])
                previous_action = np.stack(minibatch[:, 2])
                action = np.stack(minibatch[:, 3])
                reward = np.stack(minibatch[:, 4])
                done = np.stack(minibatch[:, 5])

                loss, td_error = learner.distributed_train(
                                                    state=state,
                                                    next_state=next_state,
                                                    previous_action=previous_action,
                                                    action=action,
                                                    reward=reward,
                                                    done=done,
                                                    is_weight=is_weight)

                writer.add_scalar('data/loss', loss, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)

                if train_step % 100 == 0:
                    learner.target_to_main()
                
                for i in range(len(idxs)):
                    replay_buffer.update(idxs[i], td_error[i])

    else:

        env = gym.make(data['env'][FLAGS.task])
        local_buffer = buffer_queue.LocalBuffer(
            capacity=int(1e4))

        epsilon = 1.0
        train_step = 0
        episode = 0

        state = env.reset()
        state = (state * 255).astype(np.int32)
        previous_action = 0
        score = 0

        writer = SummaryWriter('runs/{}/actor_{}'.format(data['env'][FLAGS.task], FLAGS.task))

        while True:

            actor.parameter_sync()

            for _ in range(data['trajectory']):

                action, q_value, max_q_value = actor.get_policy_and_action(
                    state=state, previous_action=previous_action, epsilon=epsilon)

                next_state, reward, done, _ = env.step(action)
                next_state = (next_state * 255).astype(np.int32)

                score += reward

                local_buffer.append(
                    state=state, done=done,
                    reward=reward, next_state=next_state,
                    previous_action=previous_action, action=action)
                
                state = next_state
                previous_action = action

                if len(local_buffer) > 3 * data['trajectory']:
                    train_step += 1
                    sampled_data = local_buffer.sample(data['trajectory'])
                    apex_queue.append_to_queue(
                        task=FLAGS.task,
                        unrolled_state=sampled_data['state'],
                        unrolled_next_state=sampled_data['next_state'],
                        unrolled_previous_action=sampled_data['previous_action'],
                        unrolled_action=sampled_data['action'],
                        unrolled_reward=sampled_data['reward'],
                        unrolled_done=sampled_data['done'])

                if done:
                    print(episode, score, epsilon)
                    writer.add_scalar('data/epsilon', epsilon, episode)
                    writer.add_scalar('data/score', score, episode)
                    episode += 1
                    score = 0
                    epsilon = 1 / (episode * 0.05 + 1)
                    state = env.reset()
                    state = state * 255
                    previous_action = 0

if __name__ == '__main__':
    tf.app.run()