import json
import os
import time
from collections import defaultdict
from datetime import datetime
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.broker_connect import AlpacaConnect
from broker.broker_metrics import AlpacaPlatformMetrics
from broker.broker_order import AlpacaTradingBot
from data_management.data_retriever import AlpacaDataRetriever
from strategies.rebalancer import Rebalancer
from config.bt_config import *

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
        self.last_filled_avg_price_update = defaultdict(lambda: None)

    def _fetch_current_prices_volumes(self):
        """Fetch current market prices using Alpaca's API."""
        return AlpacaDataRetriever(self.api).get_last_market_data('minute', symbols)

    def _calculate_portfolio_value(self):
        """Get the current portfolio value from Alpaca."""
        return self.broker_metrics.get_portfolio_value()

    def rebalance_live_portfolio(self, date, rebalance_frequency):
        """Rebalance the live portfolio to match the final target weights."""
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
        self.db_manager.save_strategy(self.strategy_info)

        # Perform rebalancing
        asset_prices, _ = self._fetch_current_prices_volumes()
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
                    current_qty = float(positions[positions['symbol'] == symbol].qty)
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

    def _update_prices_with_average_cost(self, prices):
        """Update prices with average costs if needed"""
        average_costs = self.broker_metrics.get_last_orders()
        if average_costs.empty or average_costs['filled_at'].isna().all():
            pass
        else:
            for symbol in average_costs['symbol'].unique():
                average_cost_datetime = (
                    pd.Timestamp(average_costs[average_costs['symbol'] == symbol]['filled_at'].values[0]).tz_localize(
                        'UTC')
                    .tz_convert('Europe/Paris').floor('T').tz_localize(None))
                average_cost = round(
                    float(average_costs[average_costs['symbol'] == symbol]['filled_avg_price'].values[0]), 4)
                price_to_update = \
                prices[(prices["datetime"] == average_cost_datetime) & (prices['symbol'] == symbol)]['price'].values[0]
                if average_cost != price_to_update:
                    # replace in db
                    self.db_manager.save_average_filled_price(average_cost, average_cost_datetime, symbol)
                    print(f"Updated {symbol} price to average_cost: {average_cost} at {average_cost_datetime}")

    def _record_portfolio_state(self, date):
        """Record the portfolio state after rebalancing."""
        cash_balance = self.broker_metrics.get_portfolio_cash()

        # Get positions and prices
        try:
            positions_prices = self.broker_metrics.get_all_positions()[['symbol', 'qty', 'current_price']]
            positions = dict(zip(positions_prices['symbol'], positions_prices['qty']))
            prices = dict(zip(positions_prices['symbol'], positions_prices['current_price']))
        except Exception as e:
            positions_prices = self.broker_metrics.get_all_orders()
            positions = dict(zip(positions_prices['symbol'], positions_prices['filled_qty']))
            prices = dict(zip(positions_prices['symbol'], positions_prices['filled_avg_price']))

        # Get Volumes
        _, volumes = self._fetch_current_prices_volumes()
        # Record portfolio state in the database
        self.db_manager.save_portfolio_state(
            date,
            positions,
            prices,
            volumes,
            cash_balance
        )

    def _query_portfolio_state(self):
        (self.position_history, self.price_history, self.volume_history, self.weight_history, self.transaction_history, self.cash_history,
         self.strategy_history) = (
            self.db_manager.query_portfolio_data())
        self._update_prices_with_average_cost(pd.DataFrame(self.price_history, columns=['datetime', 'symbol', 'price']))

