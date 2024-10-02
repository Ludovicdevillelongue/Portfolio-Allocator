import os
import pandas as pd
from broker.broker_connect import AlpacaConnect
from live.live_runner import LiveAllocationRunner
from reporting.live_report import LiveDashReport


class LiveTracker:
    def __init__(self, broker, data_frequency):
        self.broker=broker
        self.data_frequency=data_frequency
    def run(self):
        # path to get config save results of optimization process
        folder_path =os.path.abspath(os.path.dirname(__file__))
        broker_config_path = os.path.join(folder_path, '../config/broker_config.yml')

        if broker=="alpaca":
            api=AlpacaConnect(broker_config_path).get_api_connection()
        else:
            api=AlpacaConnect(broker_config_path).get_api_connection()
        strategy_info=pd.read_json(os.path.join(folder_path, '../signal_generator/strategy_info.json'))

        live_allocation_runner = LiveAllocationRunner(api, broker_config_path, strategy_info,
                                           data_frequency)
        # while True:
        #     live_allocation_runner.get_live_metrics()
        #     time.sleep(60)
        dash_report = LiveDashReport(live_allocation_runner=live_allocation_runner, port=8050)
        dash_report.run_server()

if __name__ == "__main__":
    data_frequency ='day'
    broker = 'alpaca'
    live_tracker=LiveTracker(broker, data_frequency)
    live_tracker.run()