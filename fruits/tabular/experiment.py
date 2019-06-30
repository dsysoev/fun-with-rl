
import os
import glob
import pickle
import numpy as np

from environment import fruit_collection


def create_state(source_fruits, fruits, ghost, agent):

    num_fruits = len(source_fruits)
    state = [[10, 10] for _ in range(num_fruits)]

    for elem in fruits:
        indx = source_fruits.index(elem)
        state[indx] = elem

    state += ghost
    state += agent

    tabular_state = [item for sublist in state for item in sublist]

    return tabular_state


def play_and_train(env, agent, t_max=300):
    """This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward"""
    total_reward = 0.0
    env.reset()

    state_data = env.get_state()
    # fruits position
    original_fruits = np.argwhere(state_data[1, :] == 1).tolist()[:]

    state = create_state(
        source_fruits=original_fruits,
        fruits=np.argwhere(state_data[1, :] == 1).tolist(),
        ghost=np.argwhere(state_data[3, :] == 1).tolist(),
        agent=np.argwhere(state_data[2, :] == 1).tolist()
    )

    for t in range(t_max):
        action = agent.get_action(state)

        next_state_data, reward, done, _ = env.step(action)

        next_state = create_state(
            source_fruits=original_fruits,
            fruits=np.argwhere(next_state_data[1, :] == 1).tolist(),
            ghost=np.argwhere(next_state_data[3, :] == 1).tolist(),
            agent=np.argwhere(next_state_data[2, :] == 1).tolist()
        )

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    return total_reward


def train(env, agent, max_gamestep, num_games, rolling_stats=100, learning_strategy='linear', verbose=False):

    if learning_strategy == 'linear':
        epsilon_list = np.linspace(1, 0, num_games)
    else:
        epsilon_list = np.linspace(1, 0.2, num_games // 2) + \
            np.linspace(0.2, 0, num_games // 2)

    rewards_list = []
    for i, epsilon in enumerate(epsilon_list):

        agent.epsilon = epsilon
        rewards_game = play_and_train(env, agent, max_gamestep)
        rewards_list.append(rewards_game)

        if i % rolling_stats == 1 and verbose:
            print('# game {} mean reward = {:.2f} std reward = {:.2f}'.format(
                (i // rolling_stats) * rolling_stats,
                np.mean(rewards_list[-rolling_stats:]),
                np.std(rewards_list[-rolling_stats:])
            ))

    return agent, rewards_list


def save_results(agent, data, exp_type='main', root_folder='results'):

    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)

    experiments = sorted(glob.glob('./{}/experiment_*'.format(root_folder)))

    if not experiments:
        name = 'experiment_{}_0'.format(exp_type)
    else:
        *n, it = experiments[-1].split('_')
        name = "{}_{}".format("_".join(n), int(it) + 1)

    if not os.path.isdir(name):
        os.mkdir(name)

    agent_path = "{}/agent.pkl".format(name)
    with open(agent_path, 'wb') as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
    print("agent was saved ({})".format(agent_path))

    data_path = '{}/data.csv'.format(name)
    data.to_csv(data_path)
    print("score data saved ({})".format(data_path))
