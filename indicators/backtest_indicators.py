import numpy as np
import pandas as pd
import pyfolio as pf

class BacktestMetrics:
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
            cash_balance = portfolio.cash_history[date]
            portfolio_value = cash_balance + sum(
                portfolio.position_history[date][symbol] * portfolio.price_history[date][symbol]
                for symbol in portfolio.symbols
            )
            portfolio_values[date] = portfolio_value
        portfolio_pnl = portfolio_values.diff().fillna(0)
        portfolio_returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return portfolio_values, portfolio_pnl, portfolio_returns

    def compute_strategy_metrics(self, portfolio, benchmark_returns):
        metrics = {}
        metrics['cash']=pd.DataFrame.from_dict(portfolio.cash_history, orient='index', columns=['Cash_Balance'])
        metrics['positions'] = pd.DataFrame(portfolio.position_history).T
        metrics['asset_prices'] = self.close_prices[self.close_prices.index.isin(metrics['positions'].index)]
        metrics['market_values'] = metrics['positions'].multiply(metrics['asset_prices'], axis=0)
        metrics['asset_returns'] = self.asset_returns[self.asset_returns.index.isin(metrics['positions'].index)]
        metrics['asset_pnl'] = self.compute_asset_pnl(portfolio)
        metrics['portfolio_values'], metrics['portfolio_pnl'], metrics[
            'portfolio_returns'] = self.compute_portfolio_pnl_returns(portfolio)
        metrics['weights'] = pd.DataFrame(portfolio.weight_history).T
        metrics['transaction'] = pd.DataFrame(portfolio.transaction_history).set_index('date')

        # Rolling metrics
        metrics['rolling_sharpe'] = pf.timeseries.rolling_sharpe(metrics['portfolio_returns'], 126)
        metrics['rolling_beta'] = pf.timeseries.rolling_beta(metrics['portfolio_returns'], benchmark_returns, 126)
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
        metrics['drawdown_table'] = pf.timeseries.gen_drawdown_table(
            pd.Series(metrics['portfolio_returns'].values, index=pd.to_datetime(metrics['portfolio_returns'].index))
        ).dropna().applymap(lambda x: round(x, 2) if isinstance(x, (float, int)) else x)
        if metrics['drawdown_table'].empty:
            metrics['drawdown_table'] = pd.DataFrame([[None] * 5], columns=['Net drawdown in %', 'Peak date',
                                                                         'Valley date', 'Recovery date', 'Duration'])
        else:
            pass
        return metrics