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
        def to_dataframe(data, index, columns, values):
            df = pd.DataFrame(data, columns=[index, columns, values])
            return df.drop_duplicates(subset=[index, columns], keep='first').pivot(index=index, columns=columns,
                                                                                   values=values)
        # Convert all histories into DataFrames
        live_portfolio.price_history = to_dataframe(live_portfolio.price_history, 'timestamp', 'symbol', 'price')
        live_portfolio.position_history = to_dataframe(live_portfolio.position_history, 'timestamp', 'symbol',
                                                       'position')
        live_portfolio.volume_history = to_dataframe(live_portfolio.volume_history, 'timestamp', 'symbol',
                                                       'volume')
        live_portfolio.weight_history = to_dataframe(live_portfolio.weight_history, 'timestamp', 'symbol', 'weight')
        live_portfolio.transaction_history = pd.DataFrame(
            live_portfolio.transaction_history, columns=['amount', 'date', 'price', 'symbol']
        )
        live_portfolio.cash_history = pd.DataFrame(live_portfolio.cash_history,
                                                   columns=['timestamp', 'cash']).set_index('timestamp')
        live_portfolio.strategy_history=pd.DataFrame(live_portfolio.strategy_history,
                     columns=['strategy_name', 'best_params', 'best_opti_algo', 'last_rebalance_date'])

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
            self.convert_portfolio_results(self.live_portfolio)
            self.benchmark_returns = (BenchmarkPortfolio(symbols, self.live_portfolio.volume_history).
                                      compute_benchmark_returns(LiveMetrics(self.live_portfolio.price_history).compute_asset_returns()))
            portfolio_metrics = LiveMetrics(self.live_portfolio.price_history).compute_strategy_metrics(self.live_portfolio, self.benchmark_returns)
            return portfolio_metrics
