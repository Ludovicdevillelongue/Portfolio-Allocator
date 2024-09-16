import os
import time

from sklearn.ensemble import RandomForestRegressor
from alloc_test.broker.broker_connect import AlpacaConnect
from alloc_test.data_management.data_handler import DataHandler
from alloc_test.indicators.performance_indicators import Metrics
from alloc_test.live.live_runner import LiveAllocationRunner
from alloc_test.reporting.live_report import LiveDashReport
from alloc_test.strategies.strat_creator import ERC, MeanVar
from xgboost import XGBRegressor
from alloc_test.strategies.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm
from alloc_test.backtest.benchmark_portfolio import BenchmarkPortfolio
from alloc_test.backtest.backtester import Backtester
from alloc_test.data_management.data_retriever import AlpacaDataRetriever


class PortfolioAllocator:
    def __init__(self, broker, data_frequency, symbols, start_date, end_date, initial_capital):
        self.broker=broker
        self.data_frequency=data_frequency
        self.symbols=symbols
        self.start_date=start_date
        self.end_date=end_date
        self.initial_capital=initial_capital



    def run(self):
        estimation_period = 30
        dash_port = 6000
        iterations = 2
        strategies = {
            'ERC': ERC()
        }
        # Define param grids for all strategies
        param_grids = {
            'MeanVar': {
                'risk_free_rate': (0.01, 0.05),
                'target_return': (0.0001, 0.00005)
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
        optimization_algorithms = [RandomSearchAlgorithm(), GridSearchAlgorithm()]

        # path to get config save results of optimization process
        broker_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../alloc_test/config/broker_config.yml'))
        strat_opti_bt_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '../alloc_test/results/alloc_strat_opti_tester_recap.csv'))
        with open(strat_opti_bt_csv, 'w') as file:
            pass

        # Step 1: Backtest using historical data
        if broker=="alpaca":
            api=AlpacaConnect(broker_config_path).get_api_connection()
        else:
            api=AlpacaConnect(broker_config_path).get_api_connection()
        data_retriever = AlpacaDataRetriever(api)
        close_prices, volumes = data_retriever.get_historical_market_data(data_frequency, symbols, start_date, end_date)
        # Fetch splits and split ratios
        stock_split, stock_split_ratios = data_retriever.get_stock_splits(symbols, start_date, end_date, broker_config_path)

        # Adjust prices and volumes for splits
        data_handler=DataHandler()
        adjusted_close_prices = data_handler.adjust_prices_for_splits(close_prices, stock_split_ratios)
        adjusted_volumes = data_handler.adjust_volumes_for_splits(volumes, stock_split_ratios)

        # Compute returns
        asset_returns=Metrics(adjusted_close_prices).compute_asset_returns()

     # Step 2: Compute benchmark returns using the BenchmarkPortfolio
        benchmark_portfolio = BenchmarkPortfolio(symbols, adjusted_volumes)
        benchmark_returns = benchmark_portfolio.compute_benchmark_returns(asset_returns)

        # Step 3: Set up strategies and run backtests with multiple strategies
        print("Running backtests...")
        backtester = Backtester(data_handler, adjusted_close_prices, asset_returns, benchmark_returns, initial_capital,
                                strategies, estimation_period, dash_port)

        backtester.run_backtest(param_grids=param_grids, iterations=iterations,
                                optimization_algorithms=optimization_algorithms, strat_opti_bt_csv=strat_opti_bt_csv)

        # Get the best strategy
        best_strategy_name, best_sharpe = backtester.get_best_strategy()
        strategy_info={
             'strategy_name':best_strategy_name,
            'strategy_params':backtester.strategies_metrics[best_strategy_name]['best_params'],
            'opti_algo':backtester.strategies_metrics[best_strategy_name]['best_opti_algo'],
            'final_weights':backtester.strategies_metrics[best_strategy_name]['weights'].iloc[-1]
        }
        print(f"Best strategy: {best_strategy_name} with Sharpe ratio: {best_sharpe}")

        # Report the backtest results
        backtester.report_backtest()

        # Step 6: Initialize LiveTrading with the selected strategy
        print("Starting live trading...")
        live_allocation_runner = LiveAllocationRunner(api, broker_config_path, symbols, strategy_info, initial_capital,
                                           data_frequency)
        live_allocation_runner.reallocate()
        # while True:
        #     live_allocation_runner.get_live_metrics()
        #     time.sleep(60)
        dash_report = LiveDashReport(live_allocation_runner=live_allocation_runner, port=8050)
        dash_report.run_server()

if __name__ == "__main__":
    data_frequency ='day'
    broker = 'alpaca'
    symbols = ['TSLA', 'NVDA', 'MSFT', 'AAPL']
    start_date = '2024-01-01'
    end_date = '2024-09-10'
    initial_capital = 100000
    pm_runner=PortfolioAllocator(broker, data_frequency, symbols, start_date, end_date, initial_capital)
    pm_runner.run()