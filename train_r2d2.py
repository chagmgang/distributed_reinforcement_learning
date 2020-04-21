import gym
import utils

import tensorflow as tf
import numpy as np

from distributed_queue import buffer_queue
from agent import r2d2
from tensorboardX import SummaryWriter


def main():
    seq_len = 32
    lstm_size = 128

    trajectory_buffer = buffer_queue.R2D2TrajectoryBuffer(
            seq_len=seq_len)
    per_replay_buffer = buffer_queue.Memory(
            capacity=int(1e4))
    agent = r2d2.Agent(
            seq_len=seq_len,
            burn_in=16,
            input_shape=[2],
            num_action=2,
            lstm_size=lstm_size)
    agent.set_session(tf.Session())
    agent.main_to_target()

    env = gym.make('CartPole-v0')
    train_step = 0
    buffer_step = 0
    episode = 0
    score = 0
    epsilon = 1.0

    state = env.reset()
    state = [state[0], state[2]]
    previous_h = np.zeros(lstm_size)
    previous_c = np.zeros(lstm_size)

    writer = SummaryWriter()

    while True:

        trajectory_buffer.init()
        buffer_step += 1
        
        for _ in range(seq_len):
            action, q_value, h, c = agent.get_action(
                state=state, h=previous_h, c=previous_c, epsilon=epsilon)
            
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = [next_state[0], next_state[2]]
            
            trajectory_buffer.append(
                state=state,
                action=action,
                reward=reward,
                done=done,
                initial_h=previous_h,
                initial_c=previous_c)

            state = next_state
            previous_h = h
            previous_c = c

            if buffer_step > 128:
                train_step += 1
                minibatch, idxs, weight = per_replay_buffer.sample(32)
                batch_state = []
                batch_action = []
                batch_reward = []
                batch_done = []
                batch_initial_c = []
                batch_initial_h = []

                for b in minibatch:
                    batch_state.append(b[0])
                    batch_action.append(b[1])
                    batch_reward.append(b[2])
                    batch_done.append(b[3])
                    batch_initial_c.append(b[4])
                    batch_initial_h.append(b[5])

                loss, td_error = agent.train(
                    state=batch_state,
                    action=batch_action,
                    h=batch_initial_h,
                    c=batch_initial_c,
                    reward=batch_reward,
                    done=batch_done,
                    weight=weight)

                writer.add_scalar('data/loss', loss, train_step)

                for i in range(len(idxs)):
                    per_replay_buffer.update(
                        idxs[i], td_error[i])

                if train_step % 500 == 0:
                    agent.main_to_target()

            if done:
                print(episode, score, epsilon)
                writer.add_scalar('data/score', score, episode)
                writer.add_scalar('data/epsilon', epsilon, episode)
                episode += 1
                score = 0
                state = env.reset()
                state = [state[0], state[2]]
                previous_h = np.zeros(lstm_size)
                previous_c = np.zeros(lstm_size)
                epsilon = 1.0 / (0.1 * episode + 1)

        data = trajectory_buffer.extract()
        td_error = agent.get_td_error(
                state=data['state'],
                action=data['action'],
                reward=data['reward'],
                done=data['done'],
                c=data['initial_c'],
                h=data['initial_h'])
        per_replay_buffer.add(
                td_error,
                [data['state'], data['action'],
                 data['reward'], data['done'],
                 data['initial_c'], data['initial_h']])

if __name__ == '__main__':
    main()
