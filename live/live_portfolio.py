import json
import os
import time
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.broker_connect import AlpacaConnect
from broker.broker_metrics import AlpacaPlatformMetrics
from broker.broker_order import AlpacaTradingBot
from data_management.data_retriever import AlpacaDataRetriever
from strategies.rebalancer import Rebalancer

class LivePortfolio:
    def __init__(self, api, broker_config_path, strategy_info, data_frequency, db_manager):
        self.api = api
        self.broker_config_path = broker_config_path
        self.strategy_info = strategy_info
        self.data_frequency = data_frequency
        self.db_manager = db_manager

        self.weight_history = {}
        self.position_history = {}
        self.price_history = {}
        self.transaction_history = []
        self.cash_history = {}
        self.strategy_history = {}

        # Initialize broker connection and metrics
        self.broker_metrics = AlpacaPlatformMetrics(self.api, self.data_frequency)
        self.broker_orders = AlpacaTradingBot(self.api, AlpacaConnect(self.broker_config_path).get_config())

    def _fetch_current_prices(self, symbols):
        """Fetch current market prices using Alpaca's API."""
        return AlpacaDataRetriever(self.api).get_last_market_data('minute', symbols)

    def _calculate_portfolio_value(self):
        """Get the current portfolio value from Alpaca."""
        return self.broker_metrics.get_portfolio_value()

    def rebalance_live_portfolio(self, date, symbols, rebalance_frequency):
        """Rebalance the live portfolio to match the final target weights."""
        self.db_manager.save_strategy(self.strategy_info)
        self.db_manager.save_weights(date, self.strategy_info['final_weights'])

        # Initialize Rebalancer
        last_rebalance_date = datetime.strptime(self.strategy_info['last_rebalance_date'],
                                                '%Y-%m-%d').date() if 'last_rebalance_date' in self.strategy_info else None
        rebalancer = Rebalancer(self.strategy_info['strategy_name'], rebalance_frequency, last_rebalance_date)

        if not rebalancer.should_rebalance(date):
            print(f"Skipping rebalance on {date} due to {rebalance_frequency} frequency.")
            return

        self.strategy_info['last_rebalance_date'] = str(date)
        with open(os.path.join(os.path.dirname(__file__), '../signal_generator/strategy_info.json'), 'r+') as f:
            data = json.load(f)
            data.update(self.strategy_info)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        
        # Perform rebalancing
        asset_prices = self._fetch_current_prices(symbols)
        portfolio_value = self._calculate_portfolio_value()
        try:
            positions = self.broker_metrics.get_all_positions()
            positions_symbols = list(positions['symbol'])
        except KeyError:
            positions_symbols = []
        bt_symbols = list(self.strategy_info['final_weights'].keys())
        for symbol in list(set(positions_symbols + bt_symbols)):
            try:
                target_position_value = portfolio_value * self.strategy_info['final_weights'][symbol]
                target_qty = target_position_value / float(asset_prices.iloc[-1][symbol])
                try:
                    current_qty = float(positions[positions['symbol']==symbol].qty)
                except Exception:
                    current_qty = 0
                order_qty = target_qty - current_qty
                if order_qty != 0:
                    side = 'buy' if order_qty > 0 else 'sell'
                    self._execute_trade(symbol, current_qty, order_qty, side)
                    self.db_manager.save_transaction(order_qty, date, float(asset_prices.iloc[-1][symbol]), symbol)
            except KeyError:
                self.broker_orders.close_position(symbol)

    def _execute_trade(self, symbol, current_qty, order_qty, side):
        """Execute a trade and update transaction history."""
        self.broker_orders.submit_order(symbol, current_qty, order_qty, side)

    def _record_portfolio_state(self, date):
        """Record the portfolio state after rebalancing."""
        cash_balance = self.broker_metrics.get_portfolio_cash()

        # Get positions and prices
        try:
            positions_prices = self.broker_metrics.get_all_positions()[['symbol', 'qty', 'current_price']]
            positions = dict(zip(positions_prices['symbol'], positions_prices['qty']))
            prices = dict(zip(positions_prices['symbol'], positions_prices['current_price']))
        except Exception as e:
            positions_prices= self.broker_metrics.get_all_orders()
            positions = dict(zip(positions_prices['symbol'], positions_prices['filled_qty']))
            prices = dict(zip(positions_prices['symbol'], positions_prices['filled_avg_price']))

        # Record portfolio state in the database
        self.db_manager.save_portfolio_state(
            date,
            positions,
            prices,
            cash_balance
        )

    def _query_portfolio_state(self):
        (self.position_history, self.price_history, self.weight_history, self.transaction_history, self.cash_history, self.strategy_history)=(
            self.db_manager.query_portfolio_data())

