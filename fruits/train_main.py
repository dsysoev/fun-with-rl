
import pandas as pd

from environment import fruit_collection

from tabular import ai
from tabular import experiment


env = fruit_collection.FruitCollectionSmall(
    rendering=False, lives=1, game_length=200, image_saving=False)

agent = ai.SarsaAgent(
        alpha=0.01,
        epsilon=0.95,
        gamma=0.99,
        legal_actions=[0, 1, 2, 3],
        strategy='mean'
    )

agent, reward_data = experiment.train(
    env, agent, max_gamestep=200, num_games=5000000,
    rolling_stats=100, learning_strategy='linear', verbose=True)

df_reward = pd.Series(reward_data).to_frame()
df_reward.columns = ['reward']
df_reward['# game'] = ((df_reward.index.map(int) // 100 + 1) * 100).values

experiment.save_results(agent, df_reward, exp_type='main', root_folder='results')
