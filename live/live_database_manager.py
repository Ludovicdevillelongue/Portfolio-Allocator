import logging
import os
from psycopg2 import pool
import yaml


class PortfolioDatabaseManager:
    """Manages storage of portfolio states in a PostgreSQL database."""

    def __init__(self):
        folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        db_config_path = os.path.join(folder_path, 'config/db_config.yml')
        with open(db_config_path, 'r') as file:
            db_config = yaml.safe_load(file)
        self.connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dbname=db_config['db_name'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        self._create_tables()

    def _get_connection(self):
        return self.connection_pool.getconn()

    def _release_connection(self, conn):
        self.connection_pool.putconn(conn)

    def close_all_connections(self):
        self.connection_pool.closeall()

    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create tables for storing portfolio state
        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_positions (
                            timestamp TIMESTAMP,
                            symbol VARCHAR(10),
                            qty REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_prices (
                            timestamp TIMESTAMP,
                            symbol VARCHAR(10),
                            price REAL)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS portfolio_volumes (
                            timestamp TIMESTAMP,
                            symbol VARCHAR(10),
                            volume REAL)''')

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

        conn.commit()
        self._release_connection(conn)

    def save_portfolio_state(self, timestamp, positions, prices, volumes, cash_balance):
        """Store the portfolio state (positions, prices, weights, and cash) in the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Save positions
            for symbol, qty in positions.items():
                cursor.execute('''INSERT INTO portfolio_positions (timestamp, symbol, qty)
                                  VALUES (%s, %s, %s)''', (timestamp, symbol, float(qty)))

            # Save prices
            for symbol, price in prices.items():
                cursor.execute('''INSERT INTO portfolio_prices (timestamp, symbol, price)
                                  VALUES (%s, %s, %s)''', (timestamp, symbol, float(price)))

            # Save volumes
            for symbol, volume in volumes.items():
                cursor.execute('''INSERT INTO portfolio_volumes (timestamp, symbol, volume)
                                  VALUES (%s, %s, %s)''', (timestamp, symbol, float(volume)))

            # Save cash balance
            cursor.execute('''INSERT INTO portfolio_cash (timestamp, cash_balance)
                              VALUES (%s, %s)''', (timestamp, float(cash_balance)))

            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving portfolio state: {e}")
        finally:
            self._release_connection(conn)

    def save_transaction(self, amount, date, price, symbol):
        """Save transaction information."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO portfolio_transactions (amount, date, price, symbol)
                              VALUES (%s, %s, %s, %s)''', (float(amount), date, float(price), symbol))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving transaction: {e}")
        finally:
            self._release_connection(conn)

    def save_average_filled_price(self, average_cost, average_cost_datetime, symbol):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''UPDATE portfolio_prices 
                              SET price = %s 
                              WHERE timestamp = %s AND symbol = %s''',
                           (float(average_cost), average_cost_datetime, symbol))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving average filled price: {e}")
        finally:
            self._release_connection(conn)

    def save_weights(self, timestamp, weights):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for symbol, weight in weights.items():
                cursor.execute('''INSERT INTO portfolio_weights (timestamp, symbol, weight)
                                  VALUES (%s, %s, %s)''', (timestamp, symbol, float(weight)))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving weights: {e}")
        finally:
            self._release_connection(conn)

    def save_strategy(self, strategy_info):
        """Save strategy information."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO portfolio_strategy (strategy_name, strategy_params, opti_algo, last_rebalance_date)
                              VALUES (%s, %s, %s, %s)''', (strategy_info['strategy_name'], strategy_info['strategy_params'],
                                                           strategy_info['opti_algo'], str(strategy_info['last_rebalance_date'])))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving strategy: {e}")
        finally:
            self._release_connection(conn)

    def query_portfolio_data(self):
        """Query portfolio data from the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Query positions
            cursor.execute('''SELECT * FROM portfolio_positions ORDER BY timestamp ASC''')
            positions_data = cursor.fetchall()

            # Query prices
            cursor.execute('''SELECT * FROM portfolio_prices ORDER BY timestamp ASC''')
            prices_data = cursor.fetchall()

            # Query volumes
            cursor.execute('''SELECT * FROM portfolio_volumes ORDER BY timestamp ASC''')
            volumes_data = cursor.fetchall()

            # Query weights
            cursor.execute('''SELECT * FROM portfolio_weights ORDER BY timestamp ASC''')
            weights_data = cursor.fetchall()

            # Query Transaction Data
            cursor.execute('''SELECT * FROM portfolio_transactions''')
            transaction_data = cursor.fetchall()

            # Query Cash data
            cursor.execute('''SELECT * FROM portfolio_cash ORDER BY timestamp ASC''')
            cash_data = cursor.fetchall()

            # Query Strategy Data
            cursor.execute('''SELECT * FROM portfolio_strategy''')
            selected_strategy_data = cursor.fetchall()

            return positions_data, prices_data, volumes_data, weights_data, transaction_data, cash_data, selected_strategy_data
        finally:
            self._release_connection(conn)

    def delete_tables(self):
        """Query portfolio data from the database and delete tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            tables = [
                'portfolio_positions',
                'portfolio_prices',
                'portfolio_volumes',
                'portfolio_weights',
                'portfolio_cash',
                'portfolio_transactions',
                'portfolio_strategy'
            ]
            data = {}
            for table in tables:
                cursor.execute(f'SELECT * FROM {table}')
                data[table] = cursor.fetchall()
                cursor.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
            conn.commit()
            return data
        finally:
            self._release_connection(conn)

    def close(self):
        conn = self._get_connection()
        self._release_connection(conn)