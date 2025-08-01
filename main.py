import argparse
from train import run_training
from evaluate import run_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crypto RL Trader')
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    args = parser.parse_args()
    if args.mode == 'train':
        run_training()
    elif args.mode == 'eval':
        run_evaluation()
