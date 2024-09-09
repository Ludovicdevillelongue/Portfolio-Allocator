import itertools
import os
import random
import time
import pyfolio as pf
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
import gym
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3 import PPO, DDPG, TD3
from gym import spaces
from xgboost import XGBRegressor

from bt_second.optimizer import StrategyOptimizer, RandomSearchAlgorithm, GridSearchAlgorithm, \
    SimulatedAnnealingAlgorithm, GeneticAlgorithm
from reporting import PyfolioReport, DashReport

warnings.filterwarnings("ignore")


# DataHandler class to handle data fetching and preprocessing
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

    def get_tradable_symbols(self, asset_class='us_equity'):
        return [a.symbol for a in self.api.list_assets(status='active', asset_class=asset_class)]


# Portfolio class to manage portfolio data and operations
class Portfolio:
    def __init__(self, initial_capital, symbols):
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.last_positions = {symbol: 0 for symbol in symbols}
        self.cash_balance = initial_capital
        self.weight_history = {}
        self.position_history = {}
        self.price_history = {}
        self.transaction_history = []

    def rebalance_portfolio(self, allocation_strategy, current_prices, historical_returns, date):
        # Calculate portfolio value
        portfolio_value = self.cash_balance + sum(
            self.last_positions[symbol] * price for symbol, price in current_prices.items()
        )

        # Compute target weights based on the allocation strategy and historical returns
        target_weights = allocation_strategy.compute_weights(historical_returns)

        # Adjust positions to meet target weights
        for symbol, target_weight in target_weights.items():
            target_position_value = portfolio_value * target_weight
            target_position_qty = target_position_value / current_prices[symbol]

            # Calculate how much quantity to change
            current_qty = self.last_positions[symbol]

            qty_change = target_position_qty - current_qty

            # If there's a difference in quantity, execute the trade
            if qty_change != 0:
                self._execute_trade(symbol, qty_change, current_prices[symbol], date)

        # Record the updated portfolio state
        self._record_portfolio_state(date, current_prices, target_weights)

    def _execute_trade(self, symbol, qty_change, price, date):
        self.transaction_history.append({'symbol': symbol, 'amount': qty_change, 'price': price, 'date': date})
        self.last_positions[symbol] += qty_change
        self.cash_balance -= qty_change * price

    def _record_portfolio_state(self, date, current_prices, target_weights):
        self.weight_history[date] = target_weights
        self.position_history[date] = {**self.last_positions, 'cash': self.cash_balance}
        self.price_history[date] = current_prices


# BenchmarkPortfolio class to represent the benchmark for comparison
class BenchmarkPortfolio:
    def __init__(self, symbols, volumes):
        self.symbols = symbols
        self.volumes = volumes
        self.volume_weights = self._calculate_volume_weights()

    def _calculate_volume_weights(self):
        total_volumes = self.volumes.sum(axis=1)
        volume_weights = self.volumes.div(total_volumes, axis=0)
        return volume_weights

    def compute_benchmark_returns(self, asset_returns):
        aligned_weights = self.volume_weights.loc[asset_returns.index]
        weighted_returns = asset_returns * aligned_weights
        benchmark_returns = weighted_returns.sum(axis=1)
        return benchmark_returns


# Abstract AllocationStrategy class to define the interface for strategies
class AllocationStrategy(ABC):
    @abstractmethod
    def compute_weights(self, historical_returns):
        pass

class ERC(AllocationStrategy):
    def __init__(self, regularization_strength=0.0):
        self.regularization_strength = regularization_strength

    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            mri = np.dot(sigma, w) / vol
            rci = mri * w
            return np.sum((rci - rci.mean()) ** 2) + self.regularization_strength * np.sum(w ** 2)

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))

class Utility:
    def is_estimation_period_satisfied(self, historical_returns, estimation_period):
        """
        Checks if the given historical returns data has enough data to satisfy the estimation period.

        Parameters:
        historical_returns (pd.DataFrame): The historical returns data.
        estimation_period (int): The number of data points required for the estimation period.

        Returns:
        bool: True if the estimation period condition is satisfied, False otherwise.
        """
        return len(historical_returns) >= estimation_period

class StrategyRunner:
    def __init__(self, data_handler, close_prices, initial_capital, estimation_period):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.estimation_period = estimation_period
        self.asset_returns = self.data_handler.compute_returns(self.close_prices)
        self.strategy_results = {}

    def run_allocation(self, strategy_instance):
        """Run the backtest for a given strategy instance."""
        portfolio = Portfolio(self.initial_capital, self.close_prices.columns)

        for date in self.asset_returns.index:
            current_prices = self.close_prices.loc[date].to_dict()
            historical_returns = self.asset_returns.loc[:date]

            # Check if we have enough data for the estimation period
            if len(historical_returns) < self.estimation_period:
                continue

            # Rebalance the portfolio using the strategy instance
            portfolio.rebalance_portfolio(strategy_instance, current_prices, historical_returns, date)

        # Store results for the strategy
        return self._get_strategy_metrics(portfolio)
    def _calculate_asset_pnl(self, portfolio):
        asset_pnl = pd.DataFrame(index=portfolio.position_history.keys(), columns=portfolio.symbols, data=0.0)
        sorted_dates = sorted(portfolio.position_history.keys())

        for i in range(1, len(sorted_dates)):
            current_date = sorted_dates[i]
            previous_date = sorted_dates[i - 1]

            for symbol in portfolio.symbols:
                position = portfolio.position_history[previous_date][symbol]
                previous_price = portfolio.price_history[previous_date][symbol]
                current_price = portfolio.price_history[current_date][symbol]
                if position != 0:
                    price_change = current_price - previous_price
                    asset_pnl.at[current_date, symbol] = position * price_change
        return asset_pnl

    def _calculate_portfolio_pnl_returns(self, portfolio):
        portfolio_values = pd.Series(index=portfolio.position_history.keys(), dtype=float)
        sorted_dates = sorted(portfolio.position_history.keys())
        for date in sorted_dates:
            cash_balance = portfolio.position_history[date]['cash']
            portfolio_value = cash_balance + sum(
                portfolio.position_history[date][symbol] * portfolio.price_history[date][symbol]
                for symbol in portfolio.symbols
            )
            portfolio_values[date] = portfolio_value
        portfolio_pnl = portfolio_values.diff().fillna(0)
        portfolio_returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return portfolio_values, portfolio_pnl, portfolio_returns

    def _get_strategy_metrics(self, portfolio):
        asset_pnl = self._calculate_asset_pnl(portfolio)
        portfolio_values, portfolio_pnl, portfolio_returns = self._calculate_portfolio_pnl_returns(portfolio)
        weights = pd.DataFrame(portfolio.weight_history).T
        positions = pd.DataFrame(portfolio.position_history).T
        transactions = pd.DataFrame(portfolio.transaction_history).set_index('date')
        selected_close_prices = self.close_prices[self.close_prices.index.isin(positions.index)]
        selected_asset_returns = self.asset_returns[self.asset_returns.index.isin(positions.index)]
        return {
            'asset_prices':selected_close_prices,
            'asset_returns':selected_asset_returns,
            'asset_pnl': asset_pnl,
            'portfolio_values': portfolio_values,
            'portfolio_pnl': portfolio_pnl,
            'portfolio_returns': portfolio_returns,
            'weights':weights,
            'positions': positions,
            'transactions': transactions
        }



# Backtester class to perform backtesting of strategies
class Backtester:
    def __init__(self, data_handler, close_prices, initial_capital, strategies, estimation_period):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.strategies = strategies
        self.estimation_period = estimation_period
        self.asset_returns = self.data_handler.compute_returns(self.close_prices)
        self.strategy_runner=StrategyRunner(self.data_handler, self.close_prices,
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
        DashReport(self.asset_returns, self.strategy_results, benchmark_returns).run()


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



class LiveTrading:
    def __init__(self, api_key, api_secret, base_url, symbols, final_weights, initial_capital):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.symbols = symbols
        self.final_weights = final_weights  # The final weights from the backtest
        self.initial_capital = initial_capital
        self.data_handler = DataHandler(api_key, api_secret, base_url)

    def get_portfolio_value(self):
        account = self.api.get_account()
        return float(account.equity + account.cash)

    def rebalance_live_portfolio(self):
        """Rebalance the live portfolio to match the final target weights."""
        # Step 1: Fetch current market prices
        current_prices = self.data_handler.fetch_last_timestamp(self.symbols)

        # Step 2: Get current portfolio value (from Alpaca)
        portfolio_value = self.get_portfolio_value()

        # Step 3: Adjust positions to match the final weights
        for symbol, target_weight in self.final_weights.items():
            target_position_value = portfolio_value * target_weight
            target_position_qty = target_position_value / current_prices[symbol]

            # Step 4: Get the current position from Alpaca
            current_qty = self.get_current_position(symbol)

            # Step 5: Calculate the quantity change needed
            qty_change = float(target_position_qty - current_qty)

            # Step 6: Execute the trade if there's a quantity change
            if qty_change != 0:
                self.execute_trade(symbol, qty_change)

    def get_current_position(self, symbol):
        """Fetch the current position for the symbol from Alpaca."""
        try:
            position = self.api.get_position(symbol)
            return float(position.qty)
        except tradeapi.rest.APIError:
            return 0  # No position currently held

    def execute_trade(self, symbol, qty_change):
        """Submit a trade to Alpaca to adjust position."""
        side = 'buy' if qty_change > 0 else 'sell'
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=abs(qty_change),
                side=side,
                type='market',
                time_in_force='day'
            )
        except Exception as e:
            print(e)
            pass

    def run(self):
        while True:
            # Rebalance the portfolio daily (or as needed)
            self.rebalance_live_portfolio()

            # Sleep for 24 hours before the next rebalance
            time.sleep(86400)  # Sleep for 1 day


# Main function to execute the script
if __name__ == "__main__":
    # Define API keys and Alpaca base URL
    API_KEY = 'PK3V3PVBNIO87YCE20YR'
    API_SECRET = 'cJypC2614CScmhmUvgFkJAcf79iLRoKZddnMd0By'
    BASE_URL = 'https://paper-api.alpaca.markets'

    # Define symbols and initial capital
    initial_capital = 100000
    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    start_date = '2024-01-01'
    end_date = '2024-09-01'

    # Define Estimation Period (Number of Data Points Required before allocating)
    estimation_period=30
    strategies = {
        'ERC': ERC()
    }
    # Define param grids for all strategies
    param_grids = {
        'MeanVar': {
            'risk_free_rate': (0.01, 0.05),
            'target_return': (0.05, 0.15)
        },
        'ERC': {
            'regularization_strength': (0.0, 0.1)
        },
        'RLAllocator': {
            'algorithm': ['PPO', 'TD3', 'DDPG'],
            'learning_rate': (0.0001, 0.01),
            'gamma': (0.9, 0.99),
            'n_steps': (64, 512)
        },
        'MLModelAllocator': {
            'model': [RandomForestRegressor(), XGBRegressor()],
            'n_estimators': (50, 500),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3)
        }
    }

    # Select optimization algorithms to use
    optimization_algorithms = [
        RandomSearchAlgorithm(), GridSearchAlgorithm()
    ]

    # Set iterations for optimization process
    iterations = 2

    # path to save results of optimization process
    strat_opti_bt_csv = os.path.join(os.path.dirname(__file__),
                                    f'../bt_second/alloc_strat_opti_tester_recap.csv')

    # Step 1: Backtest using historical data
    data_handler = DataHandler(API_KEY, API_SECRET, BASE_URL)
    close_prices, volumes = data_handler.fetch_market_data(symbols, start_date, end_date)

    # Step 2: Compute benchmark returns using the BenchmarkPortfolio
    benchmark_portfolio = BenchmarkPortfolio(symbols, volumes)
    benchmark_returns = benchmark_portfolio.compute_benchmark_returns(data_handler.compute_returns(close_prices))

    # Step 3: Set up strategies and run backtests with multiple strategies
    print("Running backtests...")
    backtester = Backtester(data_handler, close_prices, initial_capital, strategies, estimation_period)

    backtester.run_backtest(param_grids=param_grids, iterations=iterations,
                            optimization_algorithms=optimization_algorithms, strat_opti_bt_csv=strat_opti_bt_csv)

    # Get the best strategy
    best_strategy_name, best_sharpe = backtester.get_best_strategy()
    best_weights = backtester.strategy_results[best_strategy_name]['weights'].iloc[-1]
    print(f"Best strategy: {best_strategy_name} with Sharpe ratio: {best_sharpe}")

    # Report the backtest results
    backtester.report_backtest(benchmark_returns)

    # # Step 6: Initialize LiveTrading with the selected strategy
    # print("Starting live trading...")
    # live_trader = LiveTrading(API_KEY, API_SECRET, BASE_URL, symbols, best_weights, initial_capital)
    # live_trader.run()