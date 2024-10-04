import os
import psycopg2
import yaml


class PortfolioDatabaseManager:
    """Manages storage of portfolio states in a PostgreSQL database."""

    def __init__(self):
        folder_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        db_config_path = os.path.join(folder_path, 'config/db_config.yml')
        with open(db_config_path, 'r') as file:
            db_config = yaml.safe_load(file)
        self.conn = psycopg2.connect(
            dbname=db_config['db_name'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Create tables for storing portfolio state
        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_positions (
                            timestamp TIMESTAMP,
                            symbol VARCHAR(10),
                            qty REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_prices (
                            timestamp TIMESTAMP,
                            symbol VARCHAR(10),
                            price REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_weights (
                            timestamp TIMESTAMP,
                            symbol VARCHAR(10),
                            weight REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_cash (
                            timestamp TIMESTAMP,
                            cash_balance REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_transactions (
                            amount REAL,
                            date TIMESTAMP,
                            price REAL,
                            symbol VARCHAR(10))''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_strategy (
                            strategy_name VARCHAR(50),
                            strategy_params TEXT,
                            opti_algo TEXT,
                            last_rebalance_date VARCHAR(50))''')

        self.conn.commit()

    def save_portfolio_state(self, timestamp, positions, prices, cash_balance):
        """Store the portfolio state (positions, prices, weights, and cash) in the database."""
        cursor = self.conn.cursor()

        # Save positions
        for symbol, qty in positions.items():
            cursor.execute('''INSERT INTO portfolio_positions (timestamp, symbol, qty)
                              VALUES (%s, %s, %s)''', (timestamp, symbol, qty))

        # Save prices
        for symbol, price in prices.items():
            cursor.execute('''INSERT INTO portfolio_prices (timestamp, symbol, price)
                              VALUES (%s, %s, %s)''', (timestamp, symbol, price))

        # Save cash balance
        cursor.execute('''INSERT INTO portfolio_cash (timestamp, cash_balance)
                            VALUES (%s, %s)''', (timestamp, cash_balance))

        self.conn.commit()

    def save_transaction(self, amount, date, price, symbol):
        """Save transaction information."""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO portfolio_transactions (amount, date, price, symbol)
                          VALUES (%s, %s, %s, %s)''', (amount, date, price, symbol))
        self.conn.commit()

    def save_weights(self, timestamp, weights):
        # Save weights
        cursor = self.conn.cursor()
        for symbol, weight in weights.items():
            cursor.execute('''INSERT INTO portfolio_weights (timestamp, symbol, weight)
                              VALUES (%s, %s, %s)''', (timestamp, symbol, weight))

    def save_strategy(self, strategy_info):
        """Save strategy information."""
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO portfolio_strategy (strategy_name, strategy_params, opti_algo, last_rebalance_date)
                          VALUES (%s, %s, %s, %s)''', (strategy_info['strategy_name'], strategy_info['strategy_params']
                                                                       , strategy_info['opti_algo'], str(strategy_info['last_rebalance_date'])))
        self.conn.commit()

    def query_portfolio_data(self):
        """Query portfolio data from the database."""
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

        # Query Transaction Data
        cursor.execute('''SELECT * FROM portfolio_transactions''')
        transaction_data = cursor.fetchall()

        # Query Cash data
        cursor.execute('''SELECT * FROM portfolio_cash''')
        cash_data = cursor.fetchall()

        # Query Strategy Data
        cursor.execute('''SELECT * FROM portfolio_strategy''')
        selected_strategy_data = cursor.fetchall()

        return positions_data, prices_data, weights_data, transaction_data, cash_data, selected_strategy_data

    def close(self):
        self.conn.close()
