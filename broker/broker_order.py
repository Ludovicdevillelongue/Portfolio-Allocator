import time as counter

class AlpacaTradingBot:
    def __init__(self, api, broker_config):
        self.api = api
        self.broker_config=broker_config

    def submit_order(self, symbol, current_qty, order_qty, side):
        pass

    def close_position(self, symbol):
        pass