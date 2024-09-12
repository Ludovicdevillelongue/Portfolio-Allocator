import time

from alloc_test.broker.broker_connect import AlpacaConnect
from alloc_test.broker.broker_metrics import AlpacaPlatformMetrics
from alloc_test.broker.broker_order import AlpacaTradingBot
from alloc_test.data_management.data_retriever import DataRetriever, AlpacaDataRetriever


class LiveStrategyRunner:
    def __init__(self, api,broker_config_path, symbols, final_weights,
                 initial_capital, data_frequency):
        self.api=api
        self.broker_config_path=broker_config_path
        self.symbols = symbols
        self.final_weights = final_weights
        self.initial_capital = initial_capital
        self.data_frequency=data_frequency

    def rebalance_live_portfolio(self, broker_metrics, broker_orders):

        """Rebalance the signal_generator portfolio to match the final target weights."""

        # Step 1: Fetch current market prices
        current_prices = AlpacaDataRetriever(self.api).get_last_market_data('minute', self.symbols)

        # Step 2: Get current portfolio value (from Alpaca)
        portfolio_value = broker_metrics.get_portfolio_value()
        positions_symbols=list(broker_metrics.get_all_positions()['symbol'])
        bt_symbols=list(self.final_weights.keys())
        # Step 3: Adjust positions to match the final weights
        for symbol in list(set(positions_symbols + bt_symbols)):
            try:
                target_position_value = portfolio_value * self.final_weights[symbol]
                target_qty = target_position_value / current_prices[symbol]
                current_qty = float(broker_metrics.get_symbol_position(symbol).qty)
                order_qty = target_qty - current_qty
                if order_qty != 0:
                    side = 'buy' if order_qty > 0 else 'sell'
                    broker_orders.submit_order(symbol, current_qty, order_qty, side)
            except KeyError:
                broker_orders.close_position(symbol)


    def run(self):
        broker_metrics = AlpacaPlatformMetrics(self.api, self.data_frequency)
        broker_orders=AlpacaTradingBot(self.api, AlpacaConnect(self.broker_config_path).get_config())
        while True:
            # Rebalance the portfolio daily (or as needed)
            self.rebalance_live_portfolio(broker_metrics, broker_orders)

            # Sleep for 24 hours before the next rebalance
            time.sleep(86400)  # Sleep for 1 day