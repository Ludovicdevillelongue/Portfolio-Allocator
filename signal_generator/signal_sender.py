import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from broker.broker_connect import AlpacaConnect
from data_management.data_handler import DataHandler
from indicators.backtest_indicators import BacktestMetrics
from live.live_runner import LiveAllocationRunner
from backtest.benchmark_portfolio import BenchmarkPortfolio
from backtest.backtester import Backtester
from data_management.data_retriever import AlpacaDataRetriever
from strategies.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm
from config.bt_config import *

class PortfolioAllocator:
    def __init__(self):
        self.api = AlpacaConnect(broker_config_path).get_api_connection()
        self.data_retriever = AlpacaDataRetriever(self.api)

    def run(self):
        close_prices, volumes = self.data_retriever.get_historical_market_data(data_frequency, symbols, start_date, end_date)
        stock_split, stock_split_ratios = self.data_retriever.get_stock_splits(symbols, start_date, end_date, broker_config_path)

        data_handler = DataHandler()
        adjusted_close_prices = data_handler.adjust_prices_for_splits(close_prices, stock_split_ratios)
        adjusted_volumes = data_handler.adjust_volumes_for_splits(volumes, stock_split_ratios)

        asset_returns = BacktestMetrics(adjusted_close_prices).compute_asset_returns()
        benchmark_returns = BenchmarkPortfolio(symbols, adjusted_volumes).compute_benchmark_returns(asset_returns)

        optimization_algorithms = [RandomSearchAlgorithm(), GridSearchAlgorithm()]

        print("Running backtests...")
        backtester = Backtester(data_handler, adjusted_close_prices, asset_returns, benchmark_returns, initial_capital,
                                strategies, estimation_period, bt_port, rebalance_frequency)

        backtester.run_backtest(param_grids=param_grids, iterations=iterations,
                                optimization_algorithms=optimization_algorithms, strat_opti_bt_csv=strat_opti_bt_csv)

        best_strategy_name, best_sharpe = backtester.get_best_strategy()
        strategy_info = {
            'strategy_name': best_strategy_name,
            'strategy_params': backtester.strategies_metrics[best_strategy_name]['best_params'],
            'opti_algo': backtester.strategies_metrics[best_strategy_name]['best_opti_algo'],
            'final_weights': (backtester.strategies_metrics[best_strategy_name]['weights'].iloc[-1]).to_dict(),
        }

        self.update_strategy_info(strategy_info)
        print(f"Best strategy: {best_strategy_name} with Sharpe ratio: {best_sharpe}")
        backtester.report_backtest()

        print("Starting live trading...")
        live_allocation_runner = LiveAllocationRunner(self.api, broker_config_path, strategy_info, data_frequency)
        live_allocation_runner.reallocate(symbols, rebalance_frequency)
        time.sleep(100)

    def update_strategy_info(self, strategy_info):
        if os.path.exists(strategy_info_path):
            with open(strategy_info_path, 'r') as json_file:
                existing_strategy_info = json.load(json_file)
            existing_strategy_info.update(strategy_info)
            strategy_info = existing_strategy_info

        with open(strategy_info_path, 'w') as json_file:
            json.dump(strategy_info, json_file, indent=4)

if __name__ == "__main__":
    PortfolioAllocator().run()