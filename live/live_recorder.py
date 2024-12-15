import logging
import sys
import os
import time
import pandas as pd
from broker.broker_connect import AlpacaConnect
from live.live_runner import LiveAllocationRunner
from reporting.live_report import LiveDashReport
from config.bt_config import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.INFO
)


class LiveRecorder:
    def run(self, retry_interval=2):
        logging.info("Starting LiveRecorder...")
        api=AlpacaConnect(broker_config_path).get_api_connection()
        strategy_info = None
        while strategy_info is None:
            try:
                if not os.path.exists(strategy_info_path):  # Check if the file exists
                    raise FileNotFoundError(f"File {strategy_info_path} not found.")
                logging.info("Successfully loaded strategy_info.")
                break
            except FileNotFoundError as e:
                logging.error(f"Error: {e}")
                logging.info(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
        LiveAllocationRunner(api, broker_config_path, strategy_info, data_frequency).record_live_metrics()

if __name__ == "__main__":
    live_tracker=LiveRecorder()
    live_tracker.run()