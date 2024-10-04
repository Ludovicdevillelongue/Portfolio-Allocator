import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from broker.broker_connect import AlpacaConnect
from live.live_runner import LiveAllocationRunner
from reporting.live_report import LiveDashReport
from config.bt_config import *

class LiveTracker:
    def run(self):
        api=AlpacaConnect(broker_config_path).get_api_connection()
        strategy_info=pd.read_json(strategy_info_path)
        live_allocation_runner = LiveAllocationRunner(api, broker_config_path, strategy_info,
                                           data_frequency)
        # while True:
        #     live_allocation_runner.get_live_metrics()
        #     time.sleep(60)
        dash_report = LiveDashReport(live_allocation_runner=live_allocation_runner, port=bt_port+100)
        dash_report.run_server()

if __name__ == "__main__":
    live_tracker=LiveTracker()
    live_tracker.run()