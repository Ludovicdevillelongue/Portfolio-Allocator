from abc import ABC
import pandas as pd
import requests
from alpaca_trade_api.rest import TimeFrame
from alloc_test.broker.broker_connect import AlpacaConnect, BrokerConnect


class DataRetriever(ABC):
    def __init__(self, api):
        self.api = api
    def get_historical_market_data(self, symbols, start_date, end_date, timeframe):
        pass

    def get_last_market_data(self, symbols, timeframe):
        pass

    def get_tradable_symbols(self, asset_class):
        pass

    def get_stock_splits(self, symbols, start_date, end_date, broker_config_path):
        pass
class AlpacaDataRetriever(DataRetriever):
    def __init__(self, api):
        super().__init__(api)

    def get_historical_market_data(self, data_frequency, symbols, start_date, end_date):
        data_frequency_mapping = {'day': TimeFrame.Day, 'minute': TimeFrame.Minute, 'hour': TimeFrame.Hour}
        market_data = {
            symbol: self.api.get_bars(symbol, data_frequency_mapping[data_frequency], start=start_date, end=end_date).df
            for symbol in symbols}
        data = pd.concat(market_data, axis=1)
        close_prices = data.xs('close', level=1, axis=1)
        close_prices.index = close_prices.index.date
        volumes = data.xs('volume', level=1, axis=1)
        volumes.index = volumes.index.date
        return close_prices, volumes

    def get_last_market_data(self, data_frequency, symbols):
        data_frequency_mapping = {'day': TimeFrame.Day, 'minute': TimeFrame.Minute, 'hour': TimeFrame.Hour}
        market_data = {
            symbol: self.api.get_bars(symbol, data_frequency_mapping[data_frequency], limit=1).df
            for symbol in symbols}
        data = pd.concat(market_data, axis=1)
        close_prices = data.xs('close', level=1, axis=1).ffill().iloc[-1]
        return close_prices

    def get_tradable_symbols(self, asset_class):
        return [a.symbol for a in self.api.list_assets(status='active', asset_class=asset_class)]


    def get_stock_splits(self, symbols, start_date, end_date, broker_config_path):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        stock_split = pd.DataFrame(False, index=dates, columns=symbols)
        stock_split_ratios = pd.DataFrame(1.0, index=dates, columns=symbols)
        stock_split.index = stock_split.index.date
        stock_split_ratios.index = stock_split_ratios.index.date
        date_ranges = [(i, min(i + pd.DateOffset(days=90), pd.to_datetime(end_date))) for i in
                       pd.date_range(start=start_date, end=end_date, freq='90D')]
        broker_config=AlpacaConnect(broker_config_path).get_config()
        headers = {"accept": "application/json", "APCA-API-KEY-ID": broker_config['alpaca']['api_key'],
                   "APCA-API-SECRET-KEY": broker_config['alpaca']['api_secret']}

        for symbol in symbols:
            responses = [requests.get(
                f"https://paper-api.alpaca.markets/v2/corporate_actions/announcements?ca_types=Split&since="
                f"{start.strftime('%Y-%m-%d')}&until={end.strftime('%Y-%m-%d')}&symbol={symbol}",
                headers=headers).json() for start, end in date_ranges]
            flattened_responses = [item for sublist in responses for item in sublist]

            for r in flattened_responses:
                if r['ca_type'] == 'split':
                    ex_date = pd.Timestamp(r['ex_date']).date()
                    split_ratio = float(r['new_rate'])/float(r['old_rate'])
                    stock_split.loc[ex_date, symbol] = True
                    stock_split_ratios.loc[ex_date, symbol] = split_ratio
        return stock_split, stock_split_ratios
