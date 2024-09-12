import numpy as np
import pandas as pd
class Metrics:
    def __init__(self, close_prices):
        self.close_prices = close_prices
        self.asset_returns = self.compute_asset_returns()

    def compute_asset_returns(self):
        return self.close_prices.pct_change().dropna()
    def compute_asset_pnl(self, portfolio):
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

    def compute_portfolio_pnl_returns(self, portfolio):
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

    def compute_strategy_metrics(self, portfolio):
        asset_pnl = self.compute_asset_pnl(portfolio)
        portfolio_values, portfolio_pnl, portfolio_returns = self.compute_portfolio_pnl_returns(portfolio)
        weights = pd.DataFrame(portfolio.weight_history).T
        positions = pd.DataFrame(portfolio.position_history).T
        transactions = pd.DataFrame(portfolio.transaction_history).set_index('date')
        selected_close_prices = self.close_prices[self.close_prices.index.isin(positions.index)]
        selected_asset_returns = self.asset_returns[self.asset_returns.index.isin(positions.index)]
        return {
            'asset_prices': selected_close_prices,
            'asset_returns': selected_asset_returns,
            'asset_pnl': asset_pnl,
            'portfolio_values': portfolio_values,
            'portfolio_pnl': portfolio_pnl,
            'portfolio_returns': portfolio_returns,
            'weights': weights,
            'positions': positions,
            'transactions': transactions
        }