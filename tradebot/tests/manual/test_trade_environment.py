import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

from tradebot.environments.trade_environment import TradeEnvironment
import pandas as pd
from datetime import datetime
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv


def test_trade_environment():
    # Drop csv file in tests/data
    data = pd.read_csv(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../data/btcusd.csv'))

    # print(data)
    data = data.drop(['time'], axis=1)
    n = len(data)
    split_point = int(n*.8)
    train = data.iloc[:split_point]
    test = data.iloc[split_point:]

    train_env = TradeEnvironment(train, transaction_fee=0.0026, episode_length=1000)
    train_env = DummyVecEnv([lambda: train_env])
    model = PPO2(MlpLstmPolicy, train_env, nminibatches=1)
    model.learn(total_timesteps=10000)

    test_env = TradeEnvironment(test, transaction_fee=0.0026, episode_length=1000)
    test_env = DummyVecEnv([lambda: test_env])
    obs = test_env.reset()
    done = False
    cum_rewards = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = test_env.step(action)
        print(obs, reward)
        cum_rewards += reward
        test_env.render()
    print(cum_rewards)

if __name__ == '__main__':
    test_trade_environment()