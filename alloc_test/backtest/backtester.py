import threading
import time
import numpy as np
import pandas as pd
from alloc_test.backtest.portfolio import Portfolio
from alloc_test.indicators.performance_indicators import Metrics
from alloc_test.reporting.bt_report import DashReport
from alloc_test.strategies.strat_optimizer import StrategyOptimizer
import pyfolio as pf

class StrategyRunner:
    def __init__(self, data_handler, close_prices, asset_returns, initial_capital, estimation_period):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.asset_returns=asset_returns
        self.initial_capital = initial_capital
        self.estimation_period = estimation_period
        self.strategy_results = {}

    def run_allocation(self, strategy_instance):
        """Run the backtest for a given strategy instance."""
        portfolio = Portfolio(self.initial_capital, self.close_prices.columns)

        for date in self.asset_returns.index:
            current_prices = self.close_prices.loc[date].to_dict()
            historical_returns = self.asset_returns.loc[:date]

            # Check if we have enough data_management for the estimation period
            if len(historical_returns) < self.estimation_period:
                continue

            # Rebalance the portfolio using the strategy instance
            portfolio.rebalance_portfolio(strategy_instance, current_prices, historical_returns, date)

        # Store results for the strategy
        return Metrics(self.close_prices).compute_strategy_metrics(portfolio)


class Backtester:
    def __init__(self, data_handler, close_prices, asset_returns, initial_capital, strategies, estimation_period, dash_port):
        self.data_handler = data_handler
        self.close_prices=close_prices
        self.asset_returns = asset_returns
        self.initial_capital = initial_capital
        self.strategies = strategies
        self.estimation_period = estimation_period
        self.dash_port=dash_port
        self.strategy_runner=StrategyRunner(self.data_handler, self.close_prices, self.asset_returns,
                               self.initial_capital, self.estimation_period)
        self.strategy_results = {}



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
                self.strategy_results[strategy_name]=self.strategy_runner.run_allocation(strategy_instance)
                self.strategy_results[strategy_name]['best_params']=best_params
                self.strategy_results[strategy_name]['best_opti_algo']=best_opti_algo

    def report_backtest(self, benchmark_returns):
        pass
        # for strat_name, results in self.strategy_results.items():
        #     PyfolioReport('../pyfolio_results')._generate_pyfolio_report(strat_name, results['portfolio_returns'],
        #                                                                  results['positions'], results['transactions'])
        # PyfolioReport('../pyfolio_results')._generate_heatmap(self.asset_returns)
        backtest_port = self.dash_port + 1000
        dashboard_run_server = threading.Thread(
            target=lambda: DashReport(self.asset_returns, self.strategy_results, benchmark_returns, backtest_port).run_server())
        dashboard_run_server.daemon = True  # Ensures the thread is killed when the main program exits
        dashboard_run_server.start()



    def get_best_strategy(self):
        """Find the best strategy overall after all have been backtested."""
        best_strategy = None
        best_sharpe = float('-inf')
        for strategy_name, result in self.strategy_results.items():
            sharpe_ratio = pf.timeseries.sharpe_ratio(result['portfolio_returns'])
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_strategy = strategy_name
        return best_strategy, best_sharpe
