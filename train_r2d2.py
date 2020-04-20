from agent import r2d2
from distributed_queue import buffer_queue

import tensorflow as tf
import numpy as np

import gym
import utils

from tensorboardX import SummaryWriter

def main():

    input_shape = [4]
    num_action = 2
    lstm_size = 512
    hidden_list = [256, 256]
    seq_len = 40

    agent = r2d2.Agent(
        input_shape=input_shape,
        num_action=num_action,
        seq_len=seq_len,
        lstm_size=lstm_size,
        hidden_list=hidden_list,
        learning_rate=1e-4,
        discount_factor=0.997,
        model_name='learner',
        learner_name='learner')

    agent.set_session(tf.Session())

    state_list = []
    previous_action_list = []
    previous_h_list = []
    previous_c_list = []
    action_list = []
    done_list = []
    q_value_list = []
    reward_list = []

    env = gym.make('CartPole-v0')
    state = env.reset()
    state = (state * 255).astype(np.int32)
    previous_action = 0
    previous_h = np.zeros(lstm_size)
    previous_c = np.zeros(lstm_size)

    for _ in range(seq_len):

        action, q_value, max_q_value, h, c = agent.get_policy_and_action(
                state=state, h=previous_h, c=previous_c,
                previous_action=previous_action, epsilon=0)

        next_state, reward, done, _ = env.step(action)
        next_state = (next_state * 255).astype(np.int32)

        state_list.append(state)
        previous_action_list.append(previous_action)
        previous_h_list.append(previous_h)
        previous_c_list.append(previous_c)
        action_list.append(action)
        done_list.append(done)
        reward_list.append(reward)
        q_value_list.append(q_value)

        state = next_state
        previous_h = h
        previous_c = c
        previous_action = action

        if done:
            state = env.reset()
            state = (state * 255).astype(np.int32)
            previous_action = 0
            previous_h = np.zeros(lstm_size)
            previous_c = np.zeros(lstm_size)

    agent.target_main_test(
            state=np.stack([state_list]),
            previous_action=np.stack([previous_action_list]),
            initial_h=np.stack([previous_h_list]),
            initial_c=np.stack([previous_c_list]),
            action=np.stack([action_list]),
            reward=np.stack([reward_list]),
            done=np.stack([done_list]))
    agent.target_to_main()
    agent.target_main_test(
            state=np.stack([state_list]),
            previous_action=np.stack([previous_action_list]),
            initial_h=np.stack([previous_h_list]),
            initial_c=np.stack([previous_c_list]),
            action=np.stack([action_list]),
            reward=np.stack([reward_list]),
            done=np.stack([done_list]))
    print(np.stack(state_list).shape)
    print(np.stack(previous_action_list).shape)
    print(np.stack(previous_h_list).shape)
    print(np.stack(previous_c_list).shape)
    print(np.stack(action_list).shape)
    print(np.stack(done_list).shape)
    print(np.stack(q_value_list).shape)

def test():

    input_shape = [2]
    num_action = 2
    lstm_size = 512
    hidden_list = [256, 256]
    seq_len = 8

    agent = r2d2.Agent(
        input_shape=input_shape,
        num_action=num_action,
        seq_len=seq_len,
        lstm_size=lstm_size,
        hidden_list=hidden_list,
        learning_rate=1e-4,
        discount_factor=0.997,
        model_name='learner',
        learner_name='learner')

    agent.set_session(tf.Session())

    replay_buffer = buffer_queue.Memory(
        capacity=int(1e3))
    
    buffer_step = 0
    train_step = 0
    agent.target_to_main()

    env = gym.make('CartPole-v0')
    episode = 0
    score = 0
    epsilon = 1.0

    state = env.reset()
    state = (state * 255).astype(np.int32)
    state = [state[0], state[2]]
    previous_action = 0
    previous_h = np.zeros(lstm_size)
    previous_c = np.zeros(lstm_size)

    writer = SummaryWriter()

    while True:

        state_list = []
        previous_action_list = []
        previous_h_list = []
        previous_c_list = []
        action_list = []
        reward_list = []
        done_list = []

        for i in range(seq_len):

            action, q_value, max_q_value, h, c = agent.get_policy_and_action(
                state=state, h=previous_h, c=previous_c,
                previous_action=previous_action, epsilon=epsilon)

            next_state, reward, done, _ = env.step(action)
            next_state = (next_state * 255).astype(np.int32)
            next_state = [next_state[0], next_state[2]]

            score += reward

            r = 0
            if done:
                if score == 200:
                    r = 1
                else:
                    r = -1

            state_list.append(state)
            previous_action_list.append(previous_action)
            previous_h_list.append(previous_h)
            previous_c_list.append(previous_c)
            action_list.append(action)
            reward_list.append(r)
            done_list.append(done)

            state = next_state
            previous_action = action
            previous_h = h
            previous_c = c

            if done:
                print(episode, score, epsilon)
                writer.add_scalar('data/score', score, episode)
                writer.add_scalar('data/epsilon', epsilon, episode)
                episode += 1
                score = 0
                epsilon = 1.0 / (0.1 * episode + 1)
                state = env.reset()
                state = (state * 255).astype(np.int32)
                state = [state[0], state[2]]
                previous_action = 0
                previous_h = np.zeros(lstm_size)
                previous_c = np.zeros(lstm_size)

            if buffer_step > 128:
                train_step += 1
                minibatch, idxs, is_weight = replay_buffer.sample(64)

                batch_state = []
                batch_previous_action = []
                batch_initial_h = []
                batch_initial_c = []
                batch_action = []
                batch_reward = []
                batch_done = []

                for batch in minibatch:
                    batch_state.append(batch[0])
                    batch_previous_action.append(batch[1])
                    batch_initial_h.append(batch[2])
                    batch_initial_c.append(batch[3])
                    batch_action.append(batch[4])
                    batch_reward.append(batch[5])
                    batch_done.append(batch[6])

                loss, td_error = agent.train(
                    state=batch_state,
                    previous_action=batch_previous_action,
                    initial_h=batch_initial_h,
                    initial_c=batch_initial_c,
                    action=batch_action,
                    reward=batch_reward,
                    done=batch_done,
                    weight=is_weight)

                writer.add_scalar('data/loss', loss, train_step)

                for i in range(len(idxs)):
                    replay_buffer.update(idxs[i], td_error[i])

                if train_step % 2500 == 0:
                    agent.target_to_main()

        state_list = np.stack(state_list)
        previous_action_list = np.stack(previous_action_list)
        previous_h_list = np.stack(previous_h_list)
        previous_c_list = np.stack(previous_c_list)
        action_list = np.stack(action_list)
        reward_list = np.stack(reward_list)
        done_list = np.stack(done_list)

        td_error = agent.get_td_error(
                    state=state_list,
                    previous_action=previous_action_list,
                    initial_h=previous_h_list,
                    initial_c=previous_c_list,
                    action=action_list,
                    reward=reward_list,
                    done=done_list)

        buffer_step += 1
        replay_buffer.add(
            error=td_error,
            sample=[
                state_list, previous_action_list,
                previous_h_list, previous_c_list,
                action_list, reward_list,
                done_list])

if __name__ == '__main__':
    test()
    # main()
