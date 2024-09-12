import os
from abc import ABC
import yaml
import alpaca_trade_api as tradeapi

class BrokerConnect(ABC):

    def __init__(self, broker_config_path):
        self.broker_config_path=broker_config_path

    def get_config(self):
        pass

    def get_api_connection(self):
        pass

class AlpacaConnect(BrokerConnect):
    def __init__(self, broker_config_path):
        # Call the parent class (BrokerConnect) initializer
        super().__init__(broker_config_path)


    def get_config(self):
        config_file = rf'{self.broker_config_path}'
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    def get_api_connection(self):
        broker_config=self.get_config()
        api_key = broker_config['alpaca']['api_key']
        api_secret = broker_config['alpaca']['api_secret']
        base_url = "https://paper-api.alpaca.markets"
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        return api



