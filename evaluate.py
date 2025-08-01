from data.market_data import MarketDataFetcher
from env.crypto_trading_env import CryptoTradingEnv
from agent.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

fetcher = MarketDataFetcher()
data = fetcher.fetch_historical(limit=500)
prices = data[['open', 'high', 'low', 'close', 'volume']].values

env = CryptoTradingEnv(prices)
agent = DQNAgent(state_dim=prices.shape[1]+2, action_dim=3)

# TODO: Load trained model weights here
state = env.reset()
rewards = []
for t in range(len(prices)-1):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    rewards.append(reward)
    if done:
        break

plt.plot(rewards)
plt.title('Evaluation rewards')
plt.show()