import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
class DataHandler:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def fetch_market_data(self, symbols, start_date, end_date, timeframe='1D'):
        timeframe_mapping = {'1D': TimeFrame.Day, '1Min': TimeFrame.Minute, '1H': TimeFrame.Hour}
        market_data = {
            symbol: self.api.get_bars(symbol, timeframe_mapping[timeframe], start=start_date, end=end_date).df
            for symbol in symbols}
        data = pd.concat(market_data, axis=1)
        close_prices = data.xs('close', level=1, axis=1)
        volumes = data.xs('volume', level=1, axis=1)
        return close_prices, volumes

    def compute_returns(self, close_prices):
        return close_prices.pct_change().dropna()

    def fetch_last_timestamp(self, symbols, timeframe='1Min'):
        timeframe_mapping = {'1D': TimeFrame.Day, '1Min': TimeFrame.Minute, '1H': TimeFrame.Hour}
        market_data = {
            symbol: self.api.get_bars(symbol, timeframe_mapping[timeframe], limit=1).df
            for symbol in symbols}
        data = pd.concat(market_data, axis=1)
        close_prices = data.xs('close', level=1, axis=1).ffill().iloc[-1]
        return close_prices
