import ccxt
import pandas as pd

class MarketDataFetcher:
    def __init__(self, symbol='BTC/USDT', exchange_name='binance', timeframe='1h'):
        self.symbol = symbol
        self.exchange = getattr(ccxt, exchange_name)()
        self.timeframe = timeframe

    def fetch_historical(self, limit=1000):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_live(self):
        ticker = self.exchange.fetch_ticker(self.symbol)
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(
                    ticker.get('timestamp', pd.Timestamp.utcnow().timestamp() * 1000), unit='ms'
                ),
                'open': ticker.get('open'),
                'high': ticker.get('high'),
                'low': ticker.get('low'),
                'close': ticker.get('close') or ticker.get('last'),
                'volume': ticker.get('baseVolume'),
            }
        ])
        return df
