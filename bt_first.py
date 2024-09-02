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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from datetime import datetime
import warnings
import dash
from dash import dcc, html
from dash import dash_table
import plotly.graph_objs as go
warnings.filterwarnings("ignore")

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


class MeanVar(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        mean_returns = historical_returns.mean().values
        cov_matrix = historical_returns.cov().values

        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_return = np.dot(weights, mean_returns)
            sharpe_ratio = portfolio_return / np.sqrt(portfolio_variance)
            return -sharpe_ratio  # maximize Sharpe Ratio

        num_assets = len(symbols)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_assets))
        initial_weights = num_assets * [1. / num_assets, ]

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(symbols, result.x))


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


class HierarchicalRiskParity(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        corr_matrix = historical_returns.corr().values
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        try:
            # Hierarchical clustering
            link = linkage(squareform(dist_matrix), 'single')
            clusters = fcluster(link, 0.5, criterion='distance')

            weights = np.zeros(len(symbols))
            cluster_allocations = {i: 1 / len(set(clusters)) for i in set(clusters)}

            for i, symbol in enumerate(symbols):
                weights[i] = cluster_allocations[clusters[i]]

            weights /= np.sum(weights)
        except Exception as e:
            weights = np.zeros(len(symbols))
        return dict(zip(symbols, weights))

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

        portfolio_pnl, portfolio_returns = self._calculate_portfolio_pnl_returns()
        positions_df = pd.DataFrame.from_dict(self.portfolio.portfolio_history, orient='index')
        transactions_df = pd.DataFrame(self.portfolio.transaction_log).set_index('date')
        # self._generate_pyfolio_report(portfolio_returns, positions_df, transactions_df)
        return self._generate_performance_report(portfolio_pnl, portfolio_returns, benchmark_returns)

    def _calculate_portfolio_pnl_returns(self):
        portfolio_values = pd.Series({
            date: self.portfolio.portfolio_history[date]['cash'] +
                  sum(self.portfolio.portfolio_history[date][symbol] * self.portfolio.price_history[date][symbol]
                      for symbol in self.portfolio.symbols)
            for date in self.portfolio.portfolio_history
        })
        portfolio_pnl = portfolio_values - self.portfolio.initial_capital
        portfolio_returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return portfolio_pnl, portfolio_returns

    def _generate_performance_report(self, portfolio_pnl, portfolio_returns, benchmark_returns):
        # Calculate basic rolling metrics
        rolling_metrics = {
            'rolling_sharpe': pf.timeseries.rolling_sharpe(portfolio_returns, 126),
            'rolling_beta': pf.timeseries.rolling_beta(portfolio_returns, benchmark_returns, 126),
            'cumulative_returns': (1 + portfolio_returns).cumprod(),
            'portfolio_pnl': portfolio_pnl
        }

        # Calculate additional performance metrics
        annual_return = pf.timeseries.annual_return(portfolio_returns)
        annual_volatility = pf.timeseries.annual_volatility(portfolio_returns)
        sharpe_ratio = pf.timeseries.sharpe_ratio(portfolio_returns)
        calmar_ratio = pf.timeseries.calmar_ratio(portfolio_returns)
        max_drawdown = pf.timeseries.max_drawdown(portfolio_returns)
        omega_ratio = pf.timeseries.omega_ratio(portfolio_returns)
        sortino_ratio = pf.timeseries.sortino_ratio(portfolio_returns)
        tail_ratio = pf.timeseries.tail_ratio(portfolio_returns)
        daily_var = pf.timeseries.value_at_risk(portfolio_returns)

        # Store all metrics in a dictionary
        performance_metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'omega_ratio': omega_ratio,
            'sortino_ratio': sortino_ratio,
            'tail_ratio': tail_ratio,
            'daily_var': daily_var
        }
        performance_metrics = {key: (0 if pd.isna(value) else value) for key, value in performance_metrics.items()}

        # Combine rolling metrics and performance metrics
        return {**rolling_metrics, **performance_metrics}

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


class ReportPlotter:
    def __init__(self, strategy_results):
        self.strategy_results = strategy_results
        self.app = dash.Dash(__name__)
        self._setup_layout()


    def _prepare_data_for_table(self):
        """
        Prepare data for the summary metrics table.
        """
        table_data = []
        for strategy, metrics in self.strategy_results.items():
            table_data.append({
                'Strategy': strategy,
                'Annual Return': round(metrics['annual_return'],2),
                'Annual Volatility': round(metrics['annual_volatility'],2),
                'Sharpe Ratio': round(metrics['sharpe_ratio'],2),
                'Calmar Ratio': round(metrics['calmar_ratio'],2),
                'Max Drawdown': round(metrics['max_drawdown'],2),
                'Omega Ratio': round(metrics['omega_ratio'],2),
                'Sortino Ratio': round(metrics['sortino_ratio'],2),
                'Tail Ratio': round(metrics['tail_ratio'],2),
                'Daily VaR': round(metrics['daily_var'],2),
            })
        return table_data

    def _prepare_data_for_plotting(self):
        """
        Prepare the data for plotting in Dash, one graph per metric, showing all strategies as different traces.
        """
        data_dict = {
            'rolling_sharpe': [],
            'rolling_beta': [],
            'cumulative_returns': [],
            'portfolio_pnl': []
        }

        for strategy, metrics in self.strategy_results.items():
            # Prepare data for each metric
            if not metrics['rolling_sharpe'].dropna().empty:
                data_dict['rolling_sharpe'].append(
                    go.Scatter(x=metrics['rolling_sharpe'].dropna().index,
                               y=metrics['rolling_sharpe'].dropna().values,
                               mode='lines',
                               name=f"{strategy} - Rolling Sharpe"))

            if not metrics['rolling_beta'].dropna().empty:
                data_dict['rolling_beta'].append(
                    go.Scatter(x=metrics['rolling_beta'].dropna().index,
                               y=metrics['rolling_beta'].dropna().values,
                               mode='lines',
                               name=f"{strategy} - Rolling Beta"))

            if not metrics['cumulative_returns'].dropna().empty:
                data_dict['cumulative_returns'].append(
                    go.Scatter(x=metrics['cumulative_returns'].dropna().index,
                               y=metrics['cumulative_returns'].dropna().values,
                               mode='lines',
                               name=f"{strategy} - Cumulative Returns"))

            if not metrics['portfolio_pnl'].dropna().empty:
                data_dict['portfolio_pnl'].append(
                    go.Scatter(x=metrics['portfolio_pnl'].dropna().index,
                               y=metrics['portfolio_pnl'].dropna().values,
                               mode='lines',
                               name=f"{strategy} - Portfolio P&L"))

        return data_dict

    def _setup_layout(self):
        """
        Setup the layout for the Dash app, one graph per metric.
        """
        # Prepare data for plotting
        data_for_plotting = self._prepare_data_for_plotting()

        self.app.layout = html.Div(children=[
            html.H1(children='Strategies Static Performance Metrics'),

            dash_table.DataTable(
                id='performance-metrics-table',
                columns=[
                    {"name": "Strategy", "id": "Strategy"},
                    {"name": "Annual Return", "id": "Annual Return"},
                    {"name": "Cumulative Returns", "id": "Cumulative Returns"},
                    {"name": "Annual Volatility", "id": "Annual Volatility"},
                    {"name": "Sharpe Ratio", "id": "Sharpe Ratio"},
                    {"name": "Calmar Ratio", "id": "Calmar Ratio"},
                    {"name": "Max Drawdown", "id": "Max Drawdown"},
                    {"name": "Omega Ratio", "id": "Omega Ratio"},
                    {"name": "Sortino Ratio", "id": "Sortino Ratio"},
                    {"name": "Tail Ratio", "id": "Tail Ratio"},
                    {"name": "Daily VaR", "id": "Daily VaR"}
                ],
                data=self._prepare_data_for_table(),
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center'},
            ),

            html.H2(children='Strategies Rolling Performance Metrics'),

            # Graph for Rolling Sharpe
            dcc.Graph(
                id='rolling-sharpe-graph',
                figure={
                    'data': data_for_plotting['rolling_sharpe'],
                    'layout': go.Layout(
                        title='Rolling Sharpe Comparison Across Strategies',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Rolling Sharpe'}
                    )
                }
            ),

            # Graph for Rolling Beta
            dcc.Graph(
                id='rolling-beta-graph',
                figure={
                    'data': data_for_plotting['rolling_beta'],
                    'layout': go.Layout(
                        title='Rolling Beta Comparison Across Strategies',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Rolling Beta'}
                    )
                }
            ),

            # Graph for Cumulative Returns
            dcc.Graph(
                id='cumulative-returns-graph',
                figure={
                    'data': data_for_plotting['cumulative_returns'],
                    'layout': go.Layout(
                        title='Cumulative Returns Comparison Across Strategies',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Cumulative Returns'}
                    )
                }
            ),

            # Graph for Portfolio PNL
            dcc.Graph(
                id='portfolio-pnl-graph',
                figure={
                    'data': data_for_plotting['portfolio_pnl'],
                    'layout': go.Layout(
                        title='Portfolio P&L Comparison Across Strategies',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Portfolio Equity'}
                    )
                }
            )
        ])

    def run(self):
        """
        Run the Dash server.
        """
        self.app.run_server(debug=True)

# Main function to execute the script
if __name__ == "__main__":
    API_KEY = 'PKJSAZR49HCCMKAQVANZ'
    API_SECRET = 'bIooVjShhhcMgZFhJ4SsSeILMOUmq7rbD0mzoCdA'
    BASE_URL = 'https://paper-api.alpaca.markets'

    initial_capital = 100000
    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    start_date = '2024-01-01'
    end_date = '2024-09-01'

    data_handler = DataHandler(API_KEY, API_SECRET, BASE_URL)
    close_prices, volumes = data_handler.fetch_market_data(symbols, start_date, end_date)

    benchmark_portfolio = BenchmarkPortfolio(symbols, volumes)
    benchmark_returns = benchmark_portfolio.compute_benchmark_returns(data_handler.compute_daily_returns(close_prices))

    strategies = [
        OneOverN(), MinVol()
    ]

    strategy_results = {}

    for strategy in strategies:
        print(f"Running backtest for {strategy.__class__.__name__}")
        portfolio = Portfolio(initial_capital=initial_capital, symbols=symbols, allocation_strategy=strategy)
        backtester = Backtester(data_handler, close_prices, portfolio, strategy.__class__.__name__)
        strategy_results[strategy.__class__.__name__] = backtester.run_backtest(benchmark_returns)

    plotter = ReportPlotter(strategy_results)
    plotter.run()