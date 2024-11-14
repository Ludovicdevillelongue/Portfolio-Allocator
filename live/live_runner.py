import os
from datetime import datetime
import time
import pandas as pd

from backtest.benchmark_portfolio import BenchmarkPortfolio
from indicators.live_indicators import LiveMetrics
from live.live_database_manager import PortfolioDatabaseManager
from live.live_portfolio import LivePortfolio
from config.bt_config import *

class LiveAllocationRunner:
    def __init__(self, api, broker_config_path, strategy_info, data_frequency):
        self.api = api
        self.broker_config_path = broker_config_path
        self.strategy_info = strategy_info
        self.data_frequency = data_frequency
        self.db_manager = PortfolioDatabaseManager()
        self.live_portfolio=LivePortfolio(self.api, self.broker_config_path, self.strategy_info,
                                      self.data_frequency, self.db_manager)

    def convert_portfolio_results(self, live_portfolio):
        df_price = pd.DataFrame(live_portfolio.price_history, columns=['timestamp', 'symbol', 'price'])
        df_price = df_price.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        close_prices = df_price.pivot(index='timestamp', columns='symbol', values='price')
        live_portfolio.price_history = {t: {s: v for _, s, v in filter(lambda x: x[0] == t,
                                                                       live_portfolio.price_history)}
                                        for t, _, _ in live_portfolio.price_history}
        live_portfolio.position_history = {t: {s: v for _, s, v in filter(lambda x: x[0] == t,
                                                                        live_portfolio.position_history)}
                                         for t, _, _ in live_portfolio.position_history}
        live_portfolio.weight_history = {t: {s: v for _, s, v in filter(lambda x: x[0] == t,
                                                                        live_portfolio.weight_history)}
                                         for t, _, _ in live_portfolio.weight_history}
        keys = ['amount', 'date', 'price', 'symbol']
        live_portfolio.transaction_history = [dict(zip(keys, values)) for values in live_portfolio.transaction_history]
        live_portfolio.cash_history = dict(live_portfolio.cash_history)
        return close_prices

    def reallocate(self, rebalance_frequency):

        """Run the live portfolio rebalancing process."""

        # Step 1: Rebalance the portfolio once
        date = datetime.now().date()
        self.live_portfolio.rebalance_live_portfolio(date, rebalance_frequency)


    def record_live_metrics(self):
        while True:
            current_datetime = time.strftime("%Y-%m-%d %H:%M")  # Fetch current timestamp
            self.live_portfolio._record_portfolio_state(current_datetime)  # Record portfolio state
            time.sleep(60)

    def get_live_metrics(self):
        while True:
            self.live_portfolio._query_portfolio_state()
            volume_history=(pd.DataFrame(self.live_portfolio.volume_history, columns=['datetime', 'symbol', 'volume']).
                            pivot(index='datetime', columns='symbol', values='volume'))
            price_history=(pd.DataFrame(self.live_portfolio.price_history, columns=['datetime', 'symbol', 'price']).
                           pivot(index='datetime', columns='symbol', values='price'))
            self.benchmark_returns = (BenchmarkPortfolio(symbols, volume_history).
                                      compute_benchmark_returns(LiveMetrics(price_history).compute_asset_returns()))
            close_prices = self.convert_portfolio_results(self.live_portfolio)
            portfolio_metrics = LiveMetrics(close_prices).compute_strategy_metrics(self.live_portfolio, self.benchmark_returns)
            portfolio_metrics['strategy_history'] = pd.DataFrame(self.live_portfolio.strategy_history,
                                                               columns=['strategy_name', 'best_params', 'best_opti_algo', 'last_rebalance_date'])
            return portfolio_metrics
