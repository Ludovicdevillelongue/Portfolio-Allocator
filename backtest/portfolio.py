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

    def rebalance_portfolio(self, target_weights, current_prices, date):
        # Calculate portfolio value
        portfolio_value = self.cash_balance + sum(
            self.last_positions[symbol] * price for symbol, price in current_prices.items()
        )

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

    def _record_portfolio_state(self, date, current_prices, target_weights=None):
        self.cash_history[date] = self.cash_balance
        
        # Calculate current weights based on current prices and positions
        total_value = self.cash_balance + sum(
            self.last_positions[symbol] * price for symbol, price in current_prices.items()
        )
        current_weights = {
            symbol: (self.last_positions[symbol] * current_prices[symbol]) / total_value
            for symbol in self.symbols
        }
        
        # Record weights (use target_weights if provided, otherwise use current_weights)
        self.weight_history[date] = target_weights if target_weights is not None else current_weights
        
        self.position_history[date] = {**self.last_positions}
        self.price_history[date] = current_prices


