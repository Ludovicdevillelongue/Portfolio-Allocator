import os
import sqlite3


class PortfolioDatabaseManager:
    """Manages storage of portfolio states in a SQLite database."""

    def __init__(self, db_path=os.path.abspath('../port_rebal/alloc_test/live/live_portfolio.db')):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Create tables for storing portfolio state
        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_positions (
                            timestamp TEXT,
                            symbol TEXT,
                            qty REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_prices (
                            timestamp TEXT,
                            symbol TEXT,
                            price REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_weights (
                            timestamp TEXT,
                            symbol TEXT,
                            weight REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_cash (
                            timestamp TEXT,
                            cash_balance REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_transactions (
                            amount REAL,
                            date TEXT,
                            price REAL,
                            symbol TEXT)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_strategy (
                            strategy_name TEXT,
                            strategy_params TEXT,
                            opti_algo TEXT)''')


        self.conn.commit()

    def save_portfolio_state(self, timestamp, positions, prices, weights, cash_balance):
        """Store the portfolio state (positions, prices, weights, and cash) in the database."""
        cursor = self.conn.cursor()

        # Save positions
        for symbol, qty in positions.items():
            cursor.execute('''INSERT INTO portfolio_positions (timestamp, symbol, qty)
                              VALUES (?, ?, ?)''', (timestamp, symbol, qty))

        # Save prices
        for symbol, price in prices.items():
            cursor.execute('''INSERT INTO portfolio_prices (timestamp, symbol, price)
                              VALUES (?, ?, ?)''', (timestamp, symbol, price))

        # Save weights
        for symbol, weight in weights.items():
            cursor.execute('''INSERT INTO portfolio_weights (timestamp, symbol, weight)
                              VALUES (?, ?, ?)''', (timestamp, symbol, weight))

        # Save cash balance
        cursor.execute('''INSERT INTO portfolio_cash (timestamp, cash_balance)
                            VALUES (?, ?)''', (timestamp, cash_balance))

        self.conn.commit()


    def save_transaction(self, amount, date, price, symbol):
        """Save transaction information."""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO portfolio_transactions (amount, date, price, symbol)
                          VALUES (?, ?, ?, ?)''', (amount, date, price, symbol))
        self.conn.commit()

    def save_strategy(self, strategy_info):
        """Save strategy information."""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO portfolio_strategy (strategy_name, strategy_params, opti_algo)
                          VALUES (?, ?, ?)''', (strategy_info['strategy_name'], strategy_info['strategy_params']
                                                                       , strategy_info['opti_algo']))
        self.conn.commit()

    def query_portfolio_data(self):
        """Query portfolio data from the database between start_date and end_date."""
        cursor = self.conn.cursor()

        # Query positions
        cursor.execute('''SELECT * FROM portfolio_positions''')
        positions_data = cursor.fetchall()

        # Query prices
        cursor.execute('''SELECT * FROM portfolio_prices''')
        prices_data = cursor.fetchall()

        # Query weights
        cursor.execute('''SELECT * FROM portfolio_weights''')
        weights_data = cursor.fetchall()

        #Query Transaction Data
        cursor.execute('''SELECT * FROM portfolio_transactions''')
        transaction_data = cursor.fetchall()

        #Query Cash data
        cursor.execute('''SELECT * FROM portfolio_cash''')
        cash_data = cursor.fetchall()

        #Query Strategy Data
        cursor.execute('''SELECT * FROM portfolio_strategy''')
        selected_strategy_data = cursor.fetchall()

        return positions_data, prices_data, weights_data, transaction_data, cash_data, selected_strategy_data

    def close(self):
        self.conn.close()
