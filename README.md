# Autonomous Crypto Trader (RL)

This project is a simple autonomous crypto trading bot based on Deep Q-Learning (DQN). It fetches historical and live crypto price data, simulates a trading environment, and trains a DQN agent to trade profitably.

## Features
- Fetches historical and live market data from Binance using ccxt
- Custom OpenAI Gym environment for trading simulation
- DQN agent implemented with PyTorch
- Training and evaluation scripts

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the agent:
```bash
python main.py --mode train
```

Evaluate the agent:
```bash
python main.py --mode eval
```

## File Structure
- `data/market_data.py`: Fetches crypto price data
- `env/crypto_trading_env.py`: Trading environment
- `agent/dqn_agent.py`: DQN agent
- `train.py`: Training loop
- `evaluate.py`: Evaluation/backtest
- `main.py`: Entrypoint

---
This is a minimal RL crypto trader prototype. Extend with more advanced RL algorithms, risk management, multi-asset support, etc.
