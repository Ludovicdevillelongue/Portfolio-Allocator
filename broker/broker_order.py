import os
import alpaca_trade_api as tradeapi
import yaml
import time as counter


class AlpacaTradingBot:
    def __init__(self, api, broker_config):
        self.api = api
        self.broker_config=broker_config

    def submit_order(self, symbol, current_qty, order_qty, side):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=abs(order_qty),
                side=side,
                type=self.broker_config['alpaca']['order_type'],
                time_in_force=self.broker_config['alpaca']['time_in_force']
            )
            print(f"Order submitted: {side} {abs(order_qty)} shares of {symbol}")

        except Exception as e:
            if abs(order_qty)==0:
                pass
            else:
                outstanding_qty = abs(order_qty)-abs(current_qty)
                self.api.submit_order(
                    symbol=symbol,
                    qty=abs(current_qty),
                    side=side,
                    type=self.broker_config['alpaca']['order_type'],
                    time_in_force=self.broker_config['alpaca']['time_in_force']
                )
                print(f"Order submitted: {side} {abs(current_qty)} share of {symbol}")
                counter.sleep(2)
                self.api.submit_order(
                    symbol=symbol,
                    qty=abs(outstanding_qty),
                    side=side,
                    type=self.broker_config['alpaca']['order_type'],
                    time_in_force=self.broker_config['alpaca']['time_in_force']
                )
                print(f"Order submitted: {side} {abs(outstanding_qty)} share of {symbol}")


    def close_position(self, symbol):
        closed_positions=self.api.close_position(symbol)
        print(f"Close position on {symbol}")
        return closed_positions