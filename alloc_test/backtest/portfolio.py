class Portfolio:
    def __init__(self, initial_capital, symbols):
        self.initial_capital = initial_capital
        self.symbols = symbols
        self.last_positions = {symbol: 0 for symbol in symbols}
        self.cash_balance = initial_capital
        self.weight_history = {}
        self.position_history = {}
        self.price_history = {}
        self.transaction_history = []
        self.cash_history = {}

    def rebalance_portfolio(self, allocation_strategy, current_prices, historical_returns, date):
        # Calculate portfolio value
        portfolio_value = self.cash_balance + sum(
            self.last_positions[symbol] * price for symbol, price in current_prices.items()
        )

        # Compute target weights based on the allocation strategy and historical returns
        target_weights = allocation_strategy.compute_weights(historical_returns)

        # Adjust positions to meet target weights
        for symbol, target_weight in target_weights.items():
            target_position_value = portfolio_value * target_weight
            target_position_qty = target_position_value / current_prices[symbol]

            # Calculate how much quantity to change
            current_qty = self.last_positions[symbol]
            qty_change = target_position_qty - current_qty

            # If there's a difference in quantity, execute the trade
            if qty_change != 0:
                self._execute_trade(symbol, qty_change, current_prices[symbol], date)

        # Record the updated portfolio state
        self._record_portfolio_state(date, current_prices, target_weights)


    def _execute_trade(self, symbol, qty_change, price, date):
        self.transaction_history.append({'symbol': symbol, 'amount': qty_change, 'price': price, 'date': date})
        self.last_positions[symbol] += qty_change
        self.cash_balance -= qty_change * price

    def _record_portfolio_state(self, date, current_prices, target_weights):
        self.cash_history[date]=self.cash_balance
        self.weight_history[date] = target_weights
        self.position_history[date] = {**self.last_positions}
        self.price_history[date] = current_prices


