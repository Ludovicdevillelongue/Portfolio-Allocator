import json
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.ensemble import RandomForestRegressor
from broker.broker_connect import AlpacaConnect
from data_management.data_handler import DataHandler
from indicators.backtest_indicators import BacktestMetrics
from live.live_runner import LiveAllocationRunner
from reporting.live_report import LiveDashReport
from strategies.strat_creator import (ERC, MeanVar, OneOverN, ReinforcementLearningAllocator, CVaROptimization, EfficientFrontier,
                        HierarchicalRiskParity, MinVol, AdvancedHierarchicalRiskParity, MaximumDivergence, MLModelAllocator)
from xgboost import XGBRegressor
from strategies.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm
from backtest.benchmark_portfolio import BenchmarkPortfolio
from backtest.backtester import Backtester
from data_management.data_retriever import AlpacaDataRetriever, FactorDataProvider
from datetime import datetime


class PortfolioAllocator:
    def __init__(self, broker, data_frequency, symbols, start_date, end_date, initial_capital, rebalance_frequency):
        self.broker = broker
        self.data_frequency = data_frequency
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

    def run(self):
        estimation_period = 30
        dash_port = 6000
        iterations = 100
        strategies = {
            'AdvancedHierarchicalRiskParity': AdvancedHierarchicalRiskParity(),
            'HierarchicalRiskParity': HierarchicalRiskParity(),
            'ERC': ERC(),
            'MeanVar': MeanVar(),
            # 'CVaROptimization': CVaROptimization(),
            # 'ReinforcementLearningAllocator': ReinforcementLearningAllocator(),
            # 'EfficientFrontier': EfficientFrontier(),
            # 'MLModelAllocator': MLModelAllocator(),
            # 'MinVol': MinVol(),
            # 'MaximumDivergence': MaximumDivergence(),
        }
        # Define param grids for all strategies
        param_grids = {
            'ERC': {
                'regularization_strength': (0.0, 0.1)
            },
            'MeanVar': {
                'risk_free_rate': (0.01, 0.05),
                'target_return': (0.05, 0.15)
            },
            'ReinforcementLearningAllocator': {
                'algorithm': ['PPO', 'TD3', 'DDPG'],
                'learning_rate': (0.0001, 0.01),
                'gamma': (0.9, 0.99),
                'n_steps': (64, 512),
                'transaction_cost': (0.0001, 0.01),
                'risk_aversion': (0.5, 2.0),
                'total_timesteps': (5000, 20000)
            },
            'CVaROptimization': {
                'confidence_level': (0.9, 0.99)
            },
            'EfficientFrontier': {
                'target_return': (0.05, 0.15)
            },
            'HierarchicalRiskParity': {
                'linkage_method': ['single', 'complete', 'average', 'ward'],
                'distance_metric': ['correlation', 'cosine', 'dice', 'euclidean', 'hamming']
            },
            'MinVol': {
                'regularization_strength': (0.0, 0.1)
            },
            'AdvancedHierarchicalRiskParity': {
                'linkage_method': ['single', 'complete', 'average', 'ward'],
                'distance_metric': ['correlation', 'cosine', 'dice', 'euclidean', 'hamming'],
                'risk_measure': ['variance', 'standard_deviation'],
                'allocation_method': ['inverse_variance', 'equal_weight'],
                'neutralize_beta': [True, False],
                'bounds': [(0, 1)]
            },
            'MaximumDivergence': {
                'regularization_strength': (0.0, 0.1)
            },
            'MLModelAllocator': {
                'model': [RandomForestRegressor(), XGBRegressor()],
                'n_estimators': (50, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'regularization_strength': (0.0, 0.1)
            }
        }

        optimization_algorithms = [RandomSearchAlgorithm(), GridSearchAlgorithm()]

        # path to get config save results of optimization process
        folder_path = os.path.abspath(os.path.dirname(__file__))
        broker_config_path = os.path.join(folder_path, '../config/broker_config.yml')
        strat_opti_bt_csv = os.path.join(folder_path, '../results/alloc_strat_opti_tester_recap.csv')
        with open(strat_opti_bt_csv, 'w') as file:
            pass

        # Step 1: Backtest using historical data
        if self.broker == "alpaca":
            api = AlpacaConnect(broker_config_path).get_api_connection()
        else:
            api = AlpacaConnect(broker_config_path).get_api_connection()
        data_retriever = AlpacaDataRetriever(api)
        close_prices, volumes = data_retriever.get_historical_market_data(self.data_frequency, self.symbols, self.start_date, self.end_date)

        # Fetch splits and split ratios
        stock_split, stock_split_ratios = data_retriever.get_stock_splits(self.symbols, self.start_date, self.end_date, broker_config_path)

        #Fetch Factors
        # factors=FactorDataProvider(self.start_date, self.end_date).get_all_factors()


        # Adjust prices and volumes for splits
        data_handler = DataHandler()
        adjusted_close_prices = data_handler.adjust_prices_for_splits(close_prices, stock_split_ratios)
        adjusted_volumes = data_handler.adjust_volumes_for_splits(volumes, stock_split_ratios)

        # Compute asset returns
        asset_returns = BacktestMetrics(adjusted_close_prices).compute_asset_returns()

        # Step 2: Compute benchmark returns using the BenchmarkPortfolio
        benchmark_portfolio = BenchmarkPortfolio(self.symbols, adjusted_volumes)
        benchmark_returns = benchmark_portfolio.compute_benchmark_returns(asset_returns)

        # Step 3: Set up strategies and run backtests with multiple strategies
        print("Running backtests...")
        backtester = Backtester(data_handler, adjusted_close_prices, asset_returns, benchmark_returns, initial_capital,
                                strategies, estimation_period, dash_port, self.rebalance_frequency)

        backtester.run_backtest(param_grids=param_grids, iterations=iterations,
                                optimization_algorithms=optimization_algorithms, strat_opti_bt_csv=strat_opti_bt_csv)

        # Get the best strategy
        best_strategy_name, best_sharpe = backtester.get_best_strategy()
        strategy_info = {
            'strategy_name': best_strategy_name,
            'strategy_params': backtester.strategies_metrics[best_strategy_name]['best_params'],
            'opti_algo': backtester.strategies_metrics[best_strategy_name]['best_opti_algo'],
            'final_weights': (backtester.strategies_metrics[best_strategy_name]['weights'].iloc[-1]).to_dict(),
        }

        strategy_info_path = os.path.join(folder_path, 'strategy_info.json')

        if os.path.exists(strategy_info_path):
            with open(strategy_info_path, 'r') as json_file:
                existing_strategy_info = json.load(json_file)
            existing_strategy_info.update(strategy_info)
            strategy_info = existing_strategy_info

        with open(strategy_info_path, 'w') as json_file:
            json.dump(strategy_info, json_file, indent=4)

        print(f"Best strategy: {best_strategy_name} with Sharpe ratio: {best_sharpe}")

        # Report the backtest results
        backtester.report_backtest()

        # Step 6: Initialize LiveTrading with the selected strategy
        print("Starting live trading...")
        live_allocation_runner = LiveAllocationRunner(api, broker_config_path, strategy_info,
                                                      self.data_frequency)
        live_allocation_runner.reallocate(self.symbols, self.rebalance_frequency)
        time.sleep(100)


if __name__ == "__main__":
    data_frequency ='day'
    broker = 'alpaca'
    symbols = ['TSLA', 'NVDA', 'MSFT', 'AAPL']
    start_date = '2020-10-01'
    end_date = '2024-09-27'
    initial_capital = 100000
    rebalance_frequency='daily'
    pm_runner=PortfolioAllocator(broker, data_frequency, symbols, start_date, end_date, initial_capital, rebalance_frequency)
    pm_runner.run()