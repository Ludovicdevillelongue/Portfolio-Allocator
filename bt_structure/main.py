from bt_structure.backtester import Backtester
from bt_structure.data_handler import DataHandler
from bt_structure.portfolio_allocator import BenchmarkPortfolio
from bt_structure.strategies import OneOverN, MinVol

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

    # Step 1: Backtest using historical data
    data_handler = DataHandler(API_KEY, API_SECRET, BASE_URL)
    close_prices, volumes = data_handler.fetch_market_data(symbols, start_date, end_date)

    # Step 2: Compute benchmark returns using the BenchmarkPortfolio
    benchmark_portfolio = BenchmarkPortfolio(symbols, volumes)
    benchmark_returns = benchmark_portfolio.compute_benchmark_returns(data_handler.compute_returns(close_prices))

    # Step 3: Set up and run backtests with multiple strategies
    print("Running backtests...")
    strategies = [OneOverN(), MinVol()]
    backtester = Backtester(data_handler, close_prices, initial_capital, strategies)
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