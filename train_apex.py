from agent import apex

import tensorflow as tf
import numpy as np

import random
import gym

from tensorboardX import SummaryWriter

def main():

    input_shape = [4]
    num_action = 2
    discount_factor = 0.99
    gradient_clip_norm = 40.0
    reward_clipping = None
    model_name = 'learner'
    learner_name = 'learner'
    start_learning_rate = 0.001
    end_learning_rate = 0.0
    learning_frame = 100000000000
    memory_size = int(1e5)

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
        learner_name=learner_name,
        memory_size=memory_size)

    sess = tf.Session()
    learner.set_session(sess)

    env = gym.make('CartPole-v0')

    episode = 0
    epsilon = 1.0
    score = 0

    state = env.reset()
    previous_action = 0

    writer = SummaryWriter()

    train_step = 0

    learner.target_to_main()

    while True:

        for _ in range(128):

            action, q_value, _ = learner.get_policy_and_action(
                state=state, previous_action=previous_action, epsilon=epsilon)

            next_state, reward, done, _ = env.step(action)

            score += reward

            learner.append_to_memory(
                state=state,
                next_state=next_state,
                previous_action=previous_action,
                action=action,
                reward=reward,
                done=done)

            buffer.append(
                [state, next_state, previous_action,
                 action, reward, done])

            state = next_state
            previous_action = action

            if done:
                print(f'episode : {episode} | score : {score} | epsilon : {epsilon}')
                writer.add_scalar('data/score', score, episode)
                writer.add_scalar('data/epsilon', epsilon, episode)
                episode += 1
                score = 0
                epsilon = 1 / (0.05 * episode + 1)
                state = env.reset()
                previous_action = 0

            if len(buffer) > 128:
                batch = random.sample(list(buffer), 32)
                state_list = [b[0] for b in batch]
                next_state_list = [b[1] for b in batch]
                previous_action_list = [b[2] for b in batch]
                action_list = [b[3] for b in batch]
                reward_list = [b[4] for b in batch]
                done_list = [b[5] for b in batch]

                loss, learning_rate = learner.train_per(32)

                writer.add_scalar('data/loss', loss, train_step)
                writer.add_scalar('data/learning_rate', learning_rate, train_step)

                if train_step % 100 == 0:
                    learner.target_to_main()

                train_step += 1

if __name__ == '__main__':
    main()