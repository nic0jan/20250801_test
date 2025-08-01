import argparse
import train
import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crypto RL Trader')
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    args = parser.parse_args()
    if args.mode == 'train':
        train
    elif args.mode == 'eval':
        evaluate
