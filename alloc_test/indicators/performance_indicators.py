import numpy as np
import pandas as pd
import pyfolio as pf
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

    def compute_strategy_metrics(self, portfolio, benchmark_returns):
        results={}
        results['positions'] = pd.DataFrame(portfolio.position_history).T
        results['asset_prices'] = self.close_prices[self.close_prices.index.isin(results['positions'].index)]
        results['asset_returns'] = self.asset_returns[self.asset_returns.index.isin(results['positions'].index)]
        results['asset_pnl'] = self.compute_asset_pnl(portfolio)
        results['portfolio_values'], results['portfolio_pnl'],  results['portfolio_returns'] = self.compute_portfolio_pnl_returns(portfolio)
        results['weights'] = pd.DataFrame(portfolio.weight_history).T
        results['transaction'] = pd.DataFrame(portfolio.transaction_history).set_index('date')


        rolling_metrics = {
            'rolling_sharpe': pf.timeseries.rolling_sharpe(results['portfolio_returns'], 126),
            'rolling_beta': pf.timeseries.rolling_beta(results['portfolio_returns'], benchmark_returns, 126),
            'portfolio_cumulative_returns': (1 + results['portfolio_returns']).cumprod() - 1,
            'weights': results['weights'], 'asset_prices':results['asset_prices'], 'positions': results['positions'],
            'portfolio_pnl': results['portfolio_pnl'], 'asset_pnl': results['asset_pnl'],
            'portfolio_cum_pnl':results['portfolio_pnl'].cumsum(),'asset_cum_pnl': results['asset_pnl'].cumsum(),
            'asset_cumulative_returns': (1 + results['asset_returns']).cumprod() - 1
        }
        performance_metrics = {
                'annual_return': pf.timeseries.annual_return(results['portfolio_returns']),
                'annual_volatility': pf.timeseries.annual_volatility(results['portfolio_returns']),
                'sharpe_ratio': pf.timeseries.sharpe_ratio(results['portfolio_returns']),
                'calmar_ratio': pf.timeseries.calmar_ratio(results['portfolio_returns']),
                'max_drawdown': pf.timeseries.max_drawdown(results['portfolio_returns']),
                'omega_ratio': pf.timeseries.omega_ratio(results['portfolio_returns']),
                'sortino_ratio': pf.timeseries.sortino_ratio(results['portfolio_returns']),
                'tail_ratio': pf.timeseries.tail_ratio(results['portfolio_returns']),
                'daily_var': pf.timeseries.value_at_risk(results['portfolio_returns'])
            }
        performance_metrics = {k: (0 if pd.isna(v) else v) for k, v in performance_metrics.items()}
        return {**rolling_metrics, **performance_metrics}