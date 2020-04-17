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

    input_shape = [4]
    num_action = 2
    discount_factor = 0.999
    gradient_clip_norm = 40.0
    reward_clipping = None
    model_name = 'learner'
    learner_name = 'learner'
    start_learning_rate = 1e-4
    end_learning_rate = 0.0
    learning_frame = int(1e8)
    batch_size = 32

    learner = apex.Agent(
        input_shape=input_shape,
        num_action=num_action,
        discount_factor=discount_factor,
        gradient_clip_norm=gradient_clip_norm,
        reward_clipping=reward_clipping,
        start_learning_rate=start_learning_rate,
        end_learning_rate=end_learning_rate,
        learning_frame=learning_frame,
        model_name=model_name,
        learner_name=learner_name)

    learner.set_session(tf.Session())
    learner.target_to_main()

    env = gym.make('CartPole-v0')
    local_buffer = buffer_queue.LocalBuffer(
        capacity=int(1e4))
    epsilon = 1.0
    train_step = 0
    episode = 0
    
    state = env.reset()
    state = state * 255
    previous_action = 0
    score = 0

    while True:

        action, q_value, max_q_value = learner.get_policy_and_action(
            state=state, previous_action=previous_action, epsilon=epsilon)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state * 255

        score += reward

        local_buffer.append(
            state=state, done=done,
            next_state=next_state,
            previous_action=previous_action,
            action=action, reward=reward)
        
        state = next_state
        previous_action = action

        if len(local_buffer) > 3 * batch_size:
            train_step += 1
            sampled_data = local_buffer.sample(batch_size)
            learner.train(
                state=sampled_data['state'],
                next_state=sampled_data['next_state'],
                previous_action=sampled_data['previous_action'],
                action=sampled_data['action'],
                reward=sampled_data['reward'],
                done=sampled_data['done'])

            if train_step % 100 == 0:
                learner.target_to_main()

        if done:
            print(episode, score, epsilon)
            episode += 1
            score = 0
            epsilon = 1 / (episode * 0.05 + 1)
            state = env.reset()
            state = state * 255
            previous_action = 0

if __name__ == '__main__':
    tf.app.run()
