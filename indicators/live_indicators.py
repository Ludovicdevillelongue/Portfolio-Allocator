import numpy as np
import pandas as pd
import pyfolio as pf
class LiveMetrics:
    def __init__(self, prices):
        self.prices = prices
        self.asset_returns = self.compute_asset_returns()

    def compute_asset_returns(self):
        return self.prices.pct_change()

    def compute_asset_pnl(self, portfolio):
        position_history = portfolio.position_history.shift(1).fillna(0)
        price_change = portfolio.price_history.diff().fillna(0)
        asset_pnl = position_history * price_change
        asset_pnl = asset_pnl.loc[(asset_pnl != 0).any(axis=1)]
        return asset_pnl

    def compute_portfolio_pnl_returns(self, portfolio):
        position_history = portfolio.position_history
        price_history = portfolio.price_history
        portfolio_values = (position_history * price_history).sum(axis=1)
        portfolio_pnl = portfolio_values.diff().fillna(0)
        portfolio_returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        return portfolio_values, portfolio_pnl, portfolio_returns

    def compute_strategy_metrics(self, portfolio, benchmark_returns):
        metrics = {}
        metrics['cash'] = portfolio.cash_history
        metrics['positions'] = portfolio.position_history
        available_dates = metrics['positions'].index
        metrics['asset_prices'] = self.prices.loc[
            available_dates, metrics['positions'].columns.intersection(self.prices.columns)]
        try:
            metrics['asset_returns'] = (self.asset_returns.loc[
                available_dates[1:], metrics['positions'].columns.intersection(self.asset_returns.columns)])
        except Exception as e:
            metrics['asset_returns']=pd.DataFrame()
        metrics['market_values'] = metrics['positions'].multiply(metrics['asset_prices'], axis=0)
        metrics['asset_pnl'] = self.compute_asset_pnl(portfolio)
        metrics['portfolio_values'], metrics['portfolio_pnl'], metrics[
            'portfolio_returns'] = self.compute_portfolio_pnl_returns(portfolio)
        metrics['weights'] = portfolio.weight_history
        try:
            metrics['transaction'] = portfolio.transaction_history
        except Exception as e:
            metrics['transaction'] = pd.DataFrame()
        # Rolling metrics
        metrics['rolling_sharpe'] = pf.timeseries.rolling_sharpe(metrics['portfolio_returns'], 30)
        metrics['rolling_beta'] = pf.timeseries.rolling_beta(metrics['portfolio_returns'], benchmark_returns, 30)
        metrics['portfolio_cumulative_returns'] = (1 + metrics['portfolio_returns']).cumprod() - 1
        metrics['portfolio_cum_pnl'] = metrics['portfolio_pnl'].cumsum()
        metrics['asset_cum_pnl'] = metrics['asset_pnl'].cumsum()
        metrics['asset_cumulative_returns'] = (1 + metrics['asset_returns']).cumprod() - 1

        # Performance metrics
        metrics['annual_return'] = pf.timeseries.annual_return(metrics['portfolio_returns'])
        metrics['annual_volatility'] = pf.timeseries.annual_volatility(metrics['portfolio_returns'])
        metrics['alpha'], metrics['beta']=pf.timeseries.alpha_beta(metrics['portfolio_returns'], benchmark_returns)
        metrics['sharpe_ratio'] = pf.timeseries.sharpe_ratio(metrics['portfolio_returns'])
        metrics['sortino_ratio'] = pf.timeseries.sortino_ratio(metrics['portfolio_returns'])
        metrics['max_drawdown'] = pf.timeseries.max_drawdown(metrics['portfolio_returns'])
        metrics['calmar_ratio'] = pf.timeseries.calmar_ratio(metrics['portfolio_returns'])
        metrics['omega_ratio'] = pf.timeseries.omega_ratio(metrics['portfolio_returns'])
        metrics['tail_ratio'] = pf.timeseries.tail_ratio(metrics['portfolio_returns'])
        metrics['daily_var'] = pf.timeseries.value_at_risk(metrics['portfolio_returns'])
        metrics['tracking_error'] = (metrics['portfolio_returns']-benchmark_returns).std()
        metrics['information_ratio'] = metrics['alpha']/metrics['tracking_error']
        metrics['drawdown_table'] = (pf.timeseries.gen_drawdown_table(metrics['portfolio_returns']).dropna()
                                     .drop_duplicates(subset='Peak date', keep='last')
                                     .applymap(lambda x: round(x, 2) if isinstance(x, (float, int)) else x))
        if metrics['drawdown_table'].empty:
            metrics['drawdown_table'] = pd.DataFrame([[None] * 5], columns=['Net drawdown in %', 'Peak date',
                                                                         'Valley date', 'Recovery date', 'Duration'])
        else:
            pass
        metrics['strategy_history'] = portfolio.strategy_history
        return metrics
