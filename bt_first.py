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
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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

class MeanVar(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        mean_returns = historical_returns.mean().values
        cov_matrix = historical_returns.cov().values

        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_return = np.dot(weights, mean_returns)
            sharpe_ratio = (portfolio_return / np.sqrt(portfolio_variance)) * np.sqrt(252)
            return -sharpe_ratio  # maximize Sharpe Ratio

        num_assets = len(symbols)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets, ]

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(symbols, result.x))


class ERC(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            mri = (np.dot(sigma, w) / vol)
            rci = mri * w
            return np.sum((rci - rci.mean()) ** 2)

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

# Backtester class to perform backtesting of strategies
class Backtester:
    def __init__(self, data_handler, close_prices, initial_capital, strategies, estimation_period):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.strategies = strategies  # List of AllocationStrategy instances (RL, ML, others)
        self.asset_returns = self.data_handler.compute_returns(self.close_prices)
        self.strategy_results = {}
        self.estimation_period = estimation_period

    def run_backtest(self):

        optimizer = Optimizer(self)  # Initialize optimizer

        for strategy in self.strategies:
            print(f"Backtesting {strategy.__class__.__name__}")
            self._backtest_strategy(strategy)

    def _backtest_strategy(self, strategy):
        """Backtest a strategy and return strat results"""
        portfolio = Portfolio(self.initial_capital, self.close_prices.columns)
        for date in self.asset_returns.index:
            current_prices = self.close_prices.loc[date].to_dict()
            historical_returns = self.asset_returns.loc[:date]
            if len(historical_returns) < self.estimation_period:
                print(f"Skipping allocation at {date}. Not enough data for the estimation period.")
            else:
                portfolio.rebalance_portfolio(strategy, current_prices, historical_returns, date)

        # Store results for the strategy
        self.strategy_results[strategy.__class__.__name__] = self._get_strategy_metrics(portfolio)

    def _get_strategy_metrics(self, portfolio):
        asset_pnl = self._calculate_asset_pnl(portfolio)
        portfolio_values, portfolio_pnl, portfolio_returns = self._calculate_portfolio_pnl_returns(portfolio)
        weights = pd.DataFrame(portfolio.weight_history).T
        positions = pd.DataFrame(portfolio.position_history).T
        transactions = pd.DataFrame(portfolio.transaction_history).set_index('date')
        return {
            'asset_prices':self.close_prices,
            'asset_returns':self.asset_returns,
            'asset_pnl': asset_pnl,
            'portfolio_values': portfolio_values,
            'portfolio_pnl': portfolio_pnl,
            'portfolio_returns': portfolio_returns,
            'weights':weights,
            'positions': positions,
            'transactions': transactions
        }

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

    def report_backtest(self, benchmark_returns):
        pass
        for strat_name, results in self.strategy_results.items():
            PyfolioReport('pyfolio_results')._generate_pyfolio_report(strat_name, results['portfolio_returns'],
                                                               results['positions'], results['transactions'])
        PyfolioReport('pyfolio_results')._generate_heatmap(self.asset_returns)
        DashReport(self.asset_returns, self.strategy_results, benchmark_returns).run()

    def get_best_strategy(self):
        """Find the best strategy overall after all have been backtested."""
        best_strategy = None
        best_sharpe = float('-inf')
        for strategy_name, result in self.strategy_results.items():
            sharpe_ratio = result['sharpe_ratio']
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_strategy = strategy_name
        return best_strategy, best_sharpe

class Optimizer:
    def __init__(self, backtester):
        self.backtester = backtester  # Pass the backtester to run backtest for each strategy

    def evaluate_and_select_best(self, strategy, candidates,  type=None):
        """Evaluate and select the best model/algorithm for RL or ML strategies."""
        candidate_names = []
        best_sharpe = float('-inf')
        best_candidate = None

        for candidate in candidates:
            if type == "RL":
                print(f"Testing RL algorithm: {candidate}")
                strategy.algorithm = candidate  # Set the RL algorithm
            elif type == "DL":
                print(f"Testing DL model: {candidate.__class__.__name__}")
                strategy.model = candidate  # Set the ML model
            elif type == "ML":
                print(f"Testing ML model: {candidate.__class__.__name__}")
                strategy.model = candidate  # Set the ML model
            elif type =='ParamOpti':
                print(f"Testing Parameters model: {candidate.__class__.__name__}")
                strategy.model = candidate

            sharpe_ratio = self.backtester._backtest_strategy(strategy)
            candidate_names.append(f"{type}_{candidate.__class__.__name__ if type == 'ML' else candidate}")
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_candidate = candidate

        print(f"Best {type} strategy: {candidate_names[-1]} with Sharpe Ratio: {best_sharpe}")
        return best_candidate, best_sharpe


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

    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'ESMV', 'HDGE', 'IPAC', 'IWML', 'KRMD', 'GINN']
    start_date = '2022-01-01'
    end_date = '2024-08-01'

    # Estimation Period (Number of Data Points Required before allocating)
    estimation_period=30

    # Step 1: Backtest using historical data
    data_handler = DataHandler(API_KEY, API_SECRET, BASE_URL)
    close_prices, volumes = data_handler.fetch_market_data(symbols, start_date, end_date)
    trading_symbols=data_handler.get_tradable_symbols()

    # Step 2: Compute benchmark returns using the BenchmarkPortfolio
    benchmark_portfolio = BenchmarkPortfolio(symbols, volumes)
    benchmark_returns = benchmark_portfolio.compute_benchmark_returns(data_handler.compute_returns(close_prices))

    # Step 3: Set up and run backtests with multiple strategies
    print("Running backtests...")
    strategies = [ERC(), MeanVar()]
    backtester = Backtester(data_handler, close_prices, initial_capital, strategies, estimation_period)
    backtester.run_backtest()
    backtester.report_backtest(benchmark_returns)

    # Step 4: Get the best strategy (e.g., based on Sharpe ratio)
    best_strategy_name, best_sharpe = backtester.get_best_strategy()
    best_weights = backtester.strategy_results[best_strategy_name]['weights'].iloc[-1]  # Latest weights
    print(f"Best strategy: {best_strategy_name} with Sharpe ratio: {best_sharpe}")

    # Step 5: Report the backtest results
    backtester.report_backtest(benchmark_returns)

    # # Step 6: Initialize LiveTrading with the selected strategy
    # print("Starting live trading...")
    # live_trader = LiveTrading(API_KEY, API_SECRET, BASE_URL, symbols, best_weights, initial_capital)
    # live_trader.run()