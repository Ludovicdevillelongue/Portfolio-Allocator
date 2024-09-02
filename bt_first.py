import os
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import pyfolio as pf
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from datetime import datetime

# DataHandler class to handle data fetching and preprocessing
class DataHandler:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def fetch_market_data(self, symbols, start_date, end_date, timeframe='1D'):
        timeframe_mapping = {'1D': TimeFrame.Day, '1Min': TimeFrame.Minute, '1H': TimeFrame.Hour}
        market_data = {symbol: self.api.get_bars(symbol, timeframe_mapping[timeframe], start=start_date, end=end_date).df for symbol in symbols}
        data = pd.concat(market_data, axis=1)
        close_prices = data.xs('close', level=1, axis=1)
        volumes = data.xs('volume', level=1, axis=1)
        return close_prices, volumes

    def compute_daily_returns(self, close_prices):
        daily_returns = close_prices.pct_change().dropna()
        return daily_returns

# Portfolio class to manage portfolio data and operations
class Portfolio:
    def __init__(self, initial_capital, symbols, allocation_strategy):
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.allocation_strategy = allocation_strategy
        self.positions = {symbol: 0 for symbol in symbols}
        self.cash_balance = initial_capital
        self.portfolio_history = {}
        self.price_history = {}
        self.transaction_log = []

    def rebalance_portfolio(self, current_prices, historical_returns, date):
        portfolio_value = self.cash_balance + sum(self.positions[symbol] * price for symbol, price in current_prices.items())
        target_weights = self.allocation_strategy.compute_weights(historical_returns)

        for symbol, target_weight in target_weights.items():
            target_position_value = portfolio_value * target_weight
            target_position_qty = target_position_value / current_prices[symbol]
            qty_change = target_position_qty - self.positions[symbol]

            self._execute_trade(symbol, qty_change, current_prices[symbol], date)

        self._record_portfolio_state(date, current_prices)

    def _execute_trade(self, symbol, qty_change, price, date):
        self.transaction_log.append({'symbol': symbol, 'amount': qty_change, 'price': price, 'date': date})
        self.positions[symbol] += qty_change
        self.cash_balance -= qty_change * price

    def _record_portfolio_state(self, date, current_prices):
        self.portfolio_history[date] = {**self.positions, 'cash': self.cash_balance}
        self.price_history[date] = current_prices

# BenchmarkPortfolio class to represent the benchmark for comparison
class BenchmarkPortfolio:
    def __init__(self, symbols, volumes):
        self.symbols = symbols
        self.volumes = volumes
        self.market_cap_weights = self._calculate_market_cap_weights()

    def _calculate_market_cap_weights(self):
        total_volumes = self.volumes.sum(axis=1)
        market_cap_weights = self.volumes.div(total_volumes, axis=0)
        return market_cap_weights

    def compute_benchmark_returns(self, daily_returns):
        aligned_weights = self.market_cap_weights.loc[daily_returns.index]
        weighted_returns = daily_returns * aligned_weights
        benchmark_returns = weighted_returns.sum(axis=1)
        return benchmark_returns

# Abstract AllocationStrategy class to define the interface for strategies
class AllocationStrategy(ABC):
    @abstractmethod
    def compute_weights(self, historical_returns):
        pass

# Implementations of different allocation strategies
class OneOverN(AllocationStrategy):
    def compute_weights(self, historical_returns):
        num_assets = len(historical_returns.columns)
        return {symbol: 1.0 / num_assets for symbol in historical_returns.columns}

class ERC(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            vol = np.sqrt(np.dot(np.dot(w, sigma), w))
            mri = (np.dot(w, sigma) / vol)
            rci = mri * w  # contribution marginale à vol du portefeuille
            a = rci[0]
            return np.sum((rci - a) ** 2)

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))

class MinVol(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            return np.sqrt(np.dot(np.dot(w, sigma), w))

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))

class MaximumDivergence(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values
        std_dev = np.sqrt(np.diag(sigma))

        def objective(w):
            std_dev_portfolio = np.sqrt(np.dot(np.dot(w, sigma), w))
            weighted_std_dev = np.dot(w, std_dev)
            divergence = weighted_std_dev / std_dev_portfolio
            return -1. * np.log(divergence.sum())

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))

# Backtester class to perform backtesting of strategies
class Backtester:
    def __init__(self, data_handler, close_prices, portfolio, strategy_name):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.portfolio = portfolio
        self.strategy_name = strategy_name

    def run_backtest(self, benchmark_returns):
        daily_returns = self.data_handler.compute_daily_returns(self.close_prices)

        for date in daily_returns.index:
            current_prices = {symbol: self.close_prices.loc[date, symbol] for symbol in self.portfolio.symbols}
            historical_returns = daily_returns.loc[:date]
            self.portfolio.rebalance_portfolio(current_prices, historical_returns, date)

        portfolio_returns = self._calculate_portfolio_returns()
        positions_df = pd.DataFrame.from_dict(self.portfolio.portfolio_history, orient='index')
        transactions_df = pd.DataFrame(self.portfolio.transaction_log).set_index('date')
        self._generate_pyfolio_report(portfolio_returns, positions_df, transactions_df)
        self._generate_performance_report(portfolio_returns, benchmark_returns)

    def _calculate_portfolio_returns(self):
        portfolio_values = pd.Series({
            date: self.portfolio.portfolio_history[date]['cash'] +
                  sum(self.portfolio.portfolio_history[date][symbol] * self.portfolio.price_history[date][symbol]
                      for symbol in self.portfolio.symbols)
            for date in self.portfolio.portfolio_history
        })
        pnl = portfolio_values - self.portfolio.initial_capital
        returns = pnl.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return returns

    def _generate_performance_report(self, returns, benchmark_returns):
        rolling_metrics = {
            'rolling_sharpe': pf.timeseries.rolling_sharpe(returns, 126),
            'rolling_beta': pf.timeseries.rolling_beta(returns, benchmark_returns, 126)
        }
        return rolling_metrics

    def _generate_pyfolio_report(self, portfolio_returns, positions, transactions, output_dir='pyfolio_results'):
        strategy_output_dir = os.path.join(output_dir, self.strategy_name)
        os.makedirs(strategy_output_dir, exist_ok=True)

        plt.figure(figsize=(15, 10), dpi=100)

        with PdfPages(f'{strategy_output_dir}/pyfolio_report.pdf') as pdf:
            pf.create_full_tear_sheet(portfolio_returns)
            pdf.savefig()
            plt.close()

        perf_stats = pf.timeseries.perf_stats(portfolio_returns)
        perf_stats.to_csv(f'{strategy_output_dir}/performance_metrics.csv')

        plt.figure(figsize=(15, 10), dpi=100)
        pf.create_returns_tear_sheet(portfolio_returns)
        plt.savefig(f'{strategy_output_dir}/returns_plot.png')
        plt.close()

        plt.figure(figsize=(15, 10), dpi=100)
        pf.create_position_tear_sheet(portfolio_returns, positions)
        plt.savefig(f'{strategy_output_dir}/positions_plot.png')
        plt.close()

        plt.figure(figsize=(15, 10), dpi=100)
        pf.create_txn_tear_sheet(portfolio_returns, positions, transactions)
        plt.savefig(f'{strategy_output_dir}/transactions_plot.png')
        plt.close()

        plt.figure(figsize=(15, 10), dpi=100)
        pf.create_interesting_times_tear_sheet(portfolio_returns)
        plt.savefig(f'{strategy_output_dir}/interesting_times_plot.png')
        plt.close()

        print(f"Pyfolio report and plots saved to '{strategy_output_dir}' directory.")




# Main function to execute the script
if __name__ == "__main__":
    API_KEY = 'PKJSAZR49HCCMKAQVANZ'
    API_SECRET = 'bIooVjShhhcMgZFhJ4SsSeILMOUmq7rbD0mzoCdA'
    BASE_URL = 'https://paper-api.alpaca.markets'

    initial_capital = 100000
    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    start_date = '2020-01-01'
    end_date = '2024-09-01'

    data_handler = DataHandler(API_KEY, API_SECRET, BASE_URL)
    close_prices, volumes = data_handler.fetch_market_data(symbols, start_date, end_date)

    benchmark_portfolio = BenchmarkPortfolio(symbols, volumes)
    benchmark_returns = benchmark_portfolio.compute_benchmark_returns(data_handler.compute_daily_returns(close_prices))

    strategies = [
        OneOverN(),
        ERC(),
        MinVol(),
        MaximumDivergence()
    ]

    strategy_results = {}

    for strategy in strategies:
        print(f"Running backtest for {strategy.__class__.__name__}")
        portfolio = Portfolio(initial_capital=initial_capital, symbols=symbols, allocation_strategy=strategy)
        backtester = Backtester(data_handler, close_prices, portfolio, strategy.__class__.__name__)
        strategy_results[strategy.__class__.__name__] = backtester.run_backtest(benchmark_returns)
