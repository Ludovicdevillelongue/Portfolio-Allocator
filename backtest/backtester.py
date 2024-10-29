import threading
import time
import numpy as np
import sys
import os

from utils.data_splitter import DataSplitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from backtest.portfolio import Portfolio
from indicators.backtest_indicators import BacktestMetrics
from reporting.bt_report import DashReport
from strategies.strat_optimizer import StrategyOptimizer
import pyfolio as pf
from strategies.rebalancer import Rebalancer
from datetime import timedelta, datetime


class StrategyRunner:
    def __init__(self, data_handler, close_prices, asset_returns, benchmark_returns, initial_capital, in_out_sample_period, rebalance_frequency):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.asset_returns=asset_returns
        self.benchmark_returns=benchmark_returns
        self.initial_capital = initial_capital
        self.in_out_sample_period = in_out_sample_period
        self.rebalance_frequency = rebalance_frequency
        self.strategy_results = {}
        self.rebalancer = None

    def run_allocation(self, strategy_instance):
        """Run the backtest for a given strategy instance."""
        portfolio = Portfolio(self.initial_capital, self.close_prices.columns)
        self.rebalancer = Rebalancer(strategy_instance, self.rebalance_frequency)

        # Initialize DataSplitter (80% in sample, 20% out sample)
        data_splitter = DataSplitter(self.in_out_sample_period*0.8, self.in_out_sample_period*0.2)

        date = self.asset_returns.index[0]

        last_rebalance = False

        while not last_rebalance:
            # Use all available data up to the current date
            historical_returns = self.asset_returns.loc[:date]

            # Check if we have enough data for the estimation period
            if (date-historical_returns.index[0]).days < self.in_out_sample_period:
                date += pd.Timedelta(days=1)
                continue

            # Check if this is the final iteration based on remaining data
            if date > self.asset_returns.index[-1]:
                last_rebalance = True

            # Split the available data into in-sample and out-sample data
            in_sample_data, out_sample_data = data_splitter.split(historical_returns, date)

            # Apply the new weights to out-sample data
            for out_sample_date in out_sample_data.index:
                # Rebalance using in-sample data
                new_weights = self.rebalancer.rebalance(out_sample_date, in_sample_data)
                prices = self.close_prices.loc[out_sample_date].to_dict()
                if new_weights is not None:
                    portfolio.rebalance_portfolio(new_weights, prices, out_sample_date)
                    portfolio._record_portfolio_state(out_sample_date, prices, new_weights)
                else:
                    portfolio._record_portfolio_state(out_sample_date, prices)

            # Move to the next date based on the out-sample data size
            date = out_sample_data.index[-1] + pd.Timedelta(days=(self.in_out_sample_period * 0.2))

        # Store results for the strategy
        return BacktestMetrics(self.close_prices).compute_strategy_metrics(portfolio, self.benchmark_returns)


class Backtester:
    def __init__(self, data_handler, close_prices, asset_returns, benchmark_returns, initial_capital, strategies,
                 in_out_sample_period, bt_port, rebalance_frequency):
        self.data_handler = data_handler
        self.close_prices=close_prices
        self.asset_returns = asset_returns
        self.benchmark_returns = benchmark_returns
        self.initial_capital = initial_capital
        self.strategies = strategies
        self.in_out_sample_period = in_out_sample_period
        self.bt_port=bt_port
        self.rebalance_frequency = rebalance_frequency
        self.strategy_runner=StrategyRunner(self.data_handler, self.close_prices, self.asset_returns, self.benchmark_returns,
                               self.initial_capital, self.in_out_sample_period, self.rebalance_frequency)
        self.strategies_metrics = {}



    def run_backtest(self, param_grids, iterations, optimization_algorithms, strat_opti_bt_csv):
        """
        Run the backtest for each strategy and optimize it using different parameter grids.
        """
        for strategy_name, strategy_instance in self.strategies.items():
            if strategy_name in param_grids:
                selected_param_grid = param_grids[strategy_name]

                optimizer = StrategyOptimizer(
                    strategy_instance=strategy_instance,
                    param_grids=selected_param_grid,
                    optimization_algorithms=optimization_algorithms,
                    iterations=iterations,
                    strategy_runner=self.strategy_runner,
                    strat_opti_bt_csv=strat_opti_bt_csv
                )

                # Optimize the strategy and find the best parameters and Sharpe ratio
                best_opti_algo, best_params = optimizer.test_all_search_types()

                # Update the strategy instance with the best parameters
                for param_name, param_value in best_params.items():
                    setattr(strategy_instance, param_name, param_value)

                # Backtest the strategy with the best parameters
                self.strategies_metrics[strategy_name]=self.strategy_runner.run_allocation(strategy_instance)
                self.strategies_metrics[strategy_name]['best_params']= str(best_params)
                self.strategies_metrics[strategy_name]['best_opti_algo']=best_opti_algo
                self.strategies_metrics[strategy_name]['last_rebalance_date']=str(self.strategy_runner.rebalancer.
                                                                                  last_rebalance_date)

    def get_latest_weights(self, strategy_instance, last_rebalance_date):
        self.rebalancer = Rebalancer(strategy_instance, self.rebalance_frequency, last_rebalance_date)
        latest_weights = self.rebalancer.rebalance(datetime.now().date(), self.asset_returns)
        return latest_weights

    def report_backtest(self):
        pass
        # for strat_name, results in self.strategy_results.items():
        #     PyfolioReport('../pyfolio_results')._generate_pyfolio_report(strat_name, results['portfolio_returns'],
        #                                                                  results['positions'], results['transactions'])
        # PyfolioReport('../pyfolio_results')._generate_heatmap(self.asset_returns)
        backtest_port = self.bt_port + 1000
        dashboard_run_server = threading.Thread(
            target=lambda: DashReport(self.asset_returns, self.strategies_metrics, backtest_port).run_server())
        dashboard_run_server.daemon = True  # Ensures the thread is killed when the main program exits
        dashboard_run_server.start()



    def get_best_strategy(self):
        """Find the best strategy overall after all have been backtested."""
        best_strategy = None
        best_sharpe = float('-inf')
        for strategy_name, result in self.strategies_metrics.items():
            sharpe_ratio = result['sharpe_ratio']
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_strategy = strategy_name
        return best_strategy, best_sharpe
