import gymnasium as gym
import numpy as np

class CryptoTradingEnv(gym.Env):
    def __init__(self, price_history, initial_balance=1000):
        super(CryptoTradingEnv, self).__init__()
        self.price_history = price_history
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(price_history.shape[1]+2,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long
        self.current_step = 0
        self.done = False
        self.trades = []
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([
            self.price_history[self.current_step],
            [self.balance, self.position]
        ])
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}
        price = self.price_history[self.current_step][3]  # close price
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
            self.trades.append(('buy', price))
        elif action == 2 and self.position == 1:  # Sell
            reward = price - self.entry_price
            self.balance += reward
            self.position = 0
            self.trades.append(('sell', price))
        self.current_step += 1
        if self.current_step >= len(self.price_history)-1:
            self.done = True
        return self._get_obs(), reward, self.done, {}