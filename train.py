import torch
import numpy as np
from data.market_data import MarketDataFetcher
from env.crypto_trading_env import CryptoTradingEnv
from agent.dqn_agent import DQNAgent

def run_training():
    EPOCHS = 20
    STEPS = 500

    fetcher = MarketDataFetcher()
    data = fetcher.fetch_historical(limit=STEPS)
    prices = data[['open', 'high', 'low', 'close', 'volume']].values

    env = CryptoTradingEnv(prices)
    agent = DQNAgent(state_dim=prices.shape[1]+2, action_dim=3)

    for epoch in range(EPOCHS):
        state = env.reset()
        total_reward = 0
        for t in range(STEPS-1):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_target()
        print(f"Epoch {epoch+1}, Total Reward: {total_reward:.2f}")

    # Save the trained model
    torch.save(agent.model.state_dict(), 'models/dqn_agent.pth')
    print("Training complete and model saved.")

if __name__ == '__main__':
    run_training()