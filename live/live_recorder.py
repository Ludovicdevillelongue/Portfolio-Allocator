

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from broker.broker_connect import AlpacaConnect
from live.live_runner import LiveAllocationRunner
from reporting.live_report import LiveDashReport
from config.bt_config import *

class LiveRecorder:
    def run(self, retry_interval=2):
        api=AlpacaConnect(broker_config_path).get_api_connection()
        strategy_info = None
        while strategy_info is None:
            try:
                if os.path.exists(strategy_info_path):  # Check if the file exists
                    strategy_info = pd.read_json(strategy_info_path)
                    print("Successfully loaded strategy_info.")
                else:
                    raise FileNotFoundError(f"File {strategy_info_path} not found.")
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
        LiveAllocationRunner(api, broker_config_path, strategy_info, data_frequency).record_live_metrics()

if __name__ == "__main__":
    live_tracker=LiveRecorder()
    live_tracker.run()