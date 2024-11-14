import os
from abc import ABC
import pandas as pd
import requests
from alpaca_trade_api.rest import TimeFrame
from sklearn.linear_model import LinearRegression
import pandas_datareader.data as web
from broker.broker_connect import AlpacaConnect, BrokerConnect
from datetime import datetime
import pytz


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
        market_data = self.api.get_bars(symbols, data_frequency_mapping[data_frequency], start=start_date,
                                        end=end_date).df
        close_prices = market_data.pivot_table(index=market_data.index, columns='symbol', values='close')
        close_prices.index = close_prices.index.date
        volumes = market_data.pivot_table(index=market_data.index, columns='symbol', values='volume')
        volumes.index = volumes.index.date
        return close_prices, volumes

    def get_last_market_data(self, data_frequency, symbols):
        # Query and create the DataFrame with aligned timestamps
        price_volume = pd.concat({symbol: pd.DataFrame([{
            't': datetime.strptime(bar._raw['t'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc).astimezone(
                pytz.timezone('Europe/Paris')),
            'c': bar._raw['c'],
            'v': bar._raw['v']}
        ]) for symbol, bar in self.api.get_latest_bars(symbols).items()}).reset_index(
            level=0)
        close_prices = price_volume.pivot_table(index='t', columns='level_0', values='c')
        volumes = price_volume.pivot_table(index='t', columns='level_0', values='v')
        return close_prices.ffill().bfill(), volumes.ffill().bfill()

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
        broker_config = AlpacaConnect(broker_config_path).get_config()
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
                    split_ratio = float(r['new_rate']) / float(r['old_rate'])
                    stock_split.loc[ex_date, symbol] = True
                    stock_split_ratios.loc[ex_date, symbol] = split_ratio
        return stock_split, stock_split_ratios


class FactorDataProvider:
    def __init__(self, start_date, end_date, csv_path='data_management/factors_data.csv'):
        self.start_date = start_date
        self.end_date = end_date
        self.csv_path = csv_path

    def fetch_ff_factors(self, start_date, end_date):
        # Download the Fama/French factors from Kenneth French's data library
        ff_factors = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start_date, end_date)[0]
        ff_factors = ff_factors / 100
        return ff_factors

    def fetch_mom_factors(self, start_date, end_date):
        # Download the Fama/French momentum factor from Kenneth French's data library
        mom_factor = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start_date, end_date)[0]
        mom_factor = mom_factor / 100
        return mom_factor

    def get_all_factors(self):
        if os.path.exists(self.csv_path):
            all_factors = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
            last_date = all_factors.index.max()
            if last_date >= pd.to_datetime(self.end_date):
                return all_factors.loc[:self.end_date]
            else:
                start_date = last_date + pd.Timedelta(days=1)
        else:
            all_factors = pd.DataFrame()
            start_date = self.start_date

        # Fetch new factors from the start_date to end_date
        ff_factors = self.fetch_ff_factors(start_date, self.end_date)
        mom_factors = self.fetch_mom_factors(start_date, self.end_date)

        new_factors = pd.concat([ff_factors, mom_factors], axis=1)
        all_factors = pd.concat([all_factors, new_factors])

        # Save the combined factors to a CSV file
        all_factors.to_csv(self.csv_path)

        return all_factors



