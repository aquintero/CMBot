import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pandas as pd
import os
from tradebot.environments.trade_environment import TradeEnvironment


class TradeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../data/btcusd.csv'))
        self.data = self.data.drop(['time'], axis=1)
        self.env = TradeEnvironment(self.data, starting_capital=10000, transaction_fee=0, episode_length=1000)

    def test_money_no_change(self):
        obs = self.env.reset()
        done = False
        while(not done):
            obs, reward, done, info = self.env.step(0)
            assert(reward == 0)
        assert(self.env.usd == 10000)
        assert(self.env.btc == 0)
            
    def test_money_buy(self):
        obs = self.env.reset()
        self.env.usd = 10000
        self.env.btc = 0
        done = False
        obs, reward, done, info = self.env.step(1)
        btc_delta = self.env.usd_unit/self.data.iloc[self.env.start+1]['open']
        assert(self.env.usd == 10000 - self.env.usd_unit)
        assert(self.env.btc == 0 + btc_delta)

    def test_money_sell(self):        
        obs = self.env.reset()
        self.env.btc = 10
        done = False
        obs, reward, done, info = self.env.step(2)
        usd_delta = self.env.usd_unit
        btc_delta = self.env.usd_unit/self.data.iloc[self.env.start+1]['open']
        assert(self.env.usd == 10000 + usd_delta)
        assert(self.env.btc == 10 - btc_delta)

    def test_money_null_sell(self):
        obs = self.env.reset()
        done = False
        obs, reward, done, info = self.env.step(2)
        assert(self.env.usd == 10000)
        assert(self.env.btc == 0)

    def test_money_null_buy(self):
        obs = self.env.reset()
        self.env.usd = 0
        self.env.btc = 1
        done = False
        obs, reward, done, info = self.env.step(1)
        assert(self.env.usd == 0)
        assert(self.env.btc == 1)

    def test_transaction_fee_sell(self):
        obs = self.env.reset()
        self.env.transaction_fee = 0.05
        self.env.btc = 10
        done = False
        obs, reward, done, info = self.env.step(2)
        usd_delta = self.env.usd_unit * (1-self.env.transaction_fee)
        btc_delta = (self.env.usd_unit / self.data.iloc[self.env.start+1]['open'])
        assert(self.env.usd == 10000 + usd_delta)
        assert(self.env.btc == 10 - btc_delta)

    
    def test_transaction_fee_buy(self):
        obs = self.env.reset()
        self.env.transaction_fee = 0.05
        self.env.usd = 10000
        self.env.btc = 0
        done = False
        obs, reward, done, info = self.env.step(1)
        btc_delta = (self.env.usd_unit*(1-self.env.transaction_fee))/self.data.iloc[self.env.start+1]['open']
        assert(self.env.usd == 10000 - self.env.usd_unit)
        assert(self.env.btc == btc_delta)

    def test_hold_rewards(self):
        obs = self.env.reset()
        self.env.usd = 0
        self.env.btc = 1
        done = False
        cum_rewards = 0
        while not done:
            obs, reward, done, info = self.env.step(0)
            cum_rewards += reward
        diff = self.data.iloc[self.env.start+self.env.episode_length]['close'] - self.data.iloc[self.env.start]['close']
        assert(cum_rewards == diff)
