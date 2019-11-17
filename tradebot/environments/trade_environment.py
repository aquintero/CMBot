import gym
from pandas import DataFrame
import numpy as np


class TradeEnvironment(gym.Env):
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, market_data: DataFrame, episode_length: int = 1000, starting_capital: int = 10000, transaction_fee: float = 0.0026, usd_unit: int = 100):
        self.market_data = market_data
        self.episode_length = episode_length
        self.starting_captial = starting_capital
        self.transaction_fee = transaction_fee
        self.usd_unit = usd_unit

        # usd, btc, open, high, low, close
        self.observation_space = gym.spaces.Box(
            low=0, high=1000000000, shape=(6,))
        self.action_space = gym.spaces.Discrete(3)  # hold, buy, sell

    def reset(self) -> tuple:
        self.start = np.random.randint(
            len(self.market_data)-self.episode_length)
        self.steps = 0
        self.usd = self.starting_captial
        self.btc = 0
        record = self.market_data.iloc[self.start]

        return (self.usd, self.btc, record['open'], record['high'], record['low'], record['close'])

    def step(self, action):
        pos = self.start + self.steps
        record = self.market_data.iloc[pos]
        next_record = self.market_data.iloc[pos+1]

        prev_val = self.usd + self.btc * record['close']
        if action == self.BUY:
            ammt = min(self.usd, self.usd_unit)
            self.usd -= ammt
            self.btc += (ammt * (1-self.transaction_fee)) / next_record['open']
        if action == self.SELL:
            ammt = min(self.btc, self.usd_unit / next_record['open'])
            self.btc -= ammt
            self.usd += next_record['open'] * (ammt * (1-self.transaction_fee))
        curr_val = self.usd + self.btc * next_record['close']

        reward = curr_val - prev_val
        self.steps += 1

        return (self.usd, self.btc, next_record['open'], next_record['high'], next_record['low'], next_record['close']), reward, self.steps == self.episode_length, {}

    def render(self):
        pass
