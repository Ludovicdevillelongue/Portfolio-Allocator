import pyfolio as pf
import pandas as pd
import numpy as np

from bt_structure.reporting import PyfolioReport
from bt_structure.reporting import DashReport


# Backtester class to perform backtesting of strategies
class Backtester:
    def __init__(self, data_handler, close_prices, initial_capital, strategies):
        self.data_handler = data_handler
        self.close_prices = close_prices
        self.initial_capital = initial_capital
        self.strategies = strategies
        self.asset_returns = self.data_handler.compute_returns(self.close_prices)
        self.strategy_results = {}

    def run_backtest(self):
        for strategy in self.strategies:
            print(f"Running backtest for {strategy.__class__.__name__}")
            portfolio = Portfolio(initial_capital=self.initial_capital, symbols=self.close_prices.columns,
                                  allocation_strategy=strategy)
            self._backtest_strategy(portfolio, strategy)

    def _backtest_strategy(self, portfolio, strategy):
        for date in self.asset_returns.index:
            current_prices = self.close_prices.loc[date].to_dict()
            historical_returns = self.asset_returns.loc[:date]
            portfolio.rebalance_portfolio(current_prices, historical_returns, date)

        # Store results for the strategy
        self.strategy_results[strategy.__class__.__name__] = self._get_strategy_metrics(portfolio)

    def _get_strategy_metrics(self, portfolio):
        asset_pnl = self._calculate_asset_pnl(portfolio)
        portfolio_values, portfolio_pnl, portfolio_returns = self._calculate_portfolio_pnl_returns(portfolio)
        weights = pd.DataFrame(portfolio.weight_history).T
        positions = pd.DataFrame(portfolio.position_history).T
        transactions = pd.DataFrame(portfolio.transaction_history).set_index('date')
        return {
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
        best_strategy = None
        best_sharpe = float('-inf')
        for strategy_name, result in self.strategy_results.items():
            sharpe_ratio = pf.timeseries.sharpe_ratio(result['portfolio_returns'])
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_strategy = strategy_name
        return best_strategy, best_sharpe