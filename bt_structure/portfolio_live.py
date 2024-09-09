import time
import alpaca_trade_api as tradeapi

from bt_structure.data_handler import DataHandler


class LiveTrading:
    def __init__(self, api_key, api_secret, base_url, symbols, final_weights, initial_capital):
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.symbols = symbols
        self.final_weights = final_weights  # The final weights from the backtest
        self.initial_capital = initial_capital
        self.data_handler = DataHandler(api_key, api_secret, base_url)

    def get_portfolio_value(self):
        """Fetch the current account equity for live trading from Alpaca."""
        account = self.api.get_account()
        return float(account.equity)

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
