import os

import numpy as np
from data.market_data import MarketDataFetcher
from env.crypto_trading_env import CryptoTradingEnv
from agent.dqn_agent import DQNAgent

# Use a non-interactive backend if no display is available
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCHS = 20
STEPS = 500

fetcher = MarketDataFetcher()
data = fetcher.fetch_historical(limit=STEPS)
prices = data[["open", "high", "low", "close", "volume"]].values

env = CryptoTradingEnv(prices)
agent = DQNAgent(state_dim=prices.shape[1] + 2, action_dim=3)

# Prepare live plot for training rewards
rewards = []
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Reward")
ax.set_title("Training Progress")
plt.show(block=False)

for epoch in range(EPOCHS):
    state = env.reset()
    total_reward = 0
    for t in range(STEPS - 1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
        if done:
            break
    agent.update_target()
    rewards.append(total_reward)
    line.set_xdata(range(1, len(rewards) + 1))
    line.set_ydata(rewards)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
    print(f"Epoch {epoch + 1}, Total Reward: {total_reward:.2f}")

plt.ioff()
plt.show()

