import os
from datetime import datetime
import time
import pandas as pd
from alloc_test.indicators.live_indicators import LiveMetrics
from alloc_test.live.live_database_manager import PortfolioDatabaseManager
from alloc_test.live.live_portfolio import LivePortfolio


class LiveAllocationRunner:
    def __init__(self, api, broker_config_path, strategy_info, data_frequency):
        self.api = api
        self.broker_config_path = broker_config_path
        self.strategy_info=strategy_info
        self.data_frequency = data_frequency

    def convert_portfolio_results(self, live_portfolio):
        df_price = pd.DataFrame(live_portfolio.price_history, columns=['timestamp', 'symbol', 'price'])
        df_price=df_price.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        close_prices = df_price.pivot(index='timestamp', columns='symbol', values='price')
        live_portfolio.price_history = {t: {s: v for _, s, v in filter(lambda x: x[0] == t,
                                                                       live_portfolio.price_history)}
                                        for t, _, _ in live_portfolio.price_history}
        live_portfolio.position_history={t: {s: v for _, s, v in filter(lambda x: x[0] == t,
                                                                        live_portfolio.position_history)}
                                         for t, _, _ in live_portfolio.position_history}
        live_portfolio.weight_history={t: {s: v for _, s, v in filter(lambda x: x[0] == t,
                                                                        live_portfolio.weight_history)}
                                         for t, _, _ in live_portfolio.weight_history}
        keys = ['amount', 'date', 'price', 'symbol']
        live_portfolio.transaction_history=[dict(zip(keys, values)) for values in live_portfolio.transaction_history]
        live_portfolio.cash_history=dict(live_portfolio.cash_history)
        return close_prices

    def reallocate(self, symbols):
        db_manager = PortfolioDatabaseManager()
        """Run the live portfolio rebalancing process."""

        liveportfolio = LivePortfolio(self.api, self.broker_config_path, self.strategy_info,
                                      self.data_frequency, db_manager)

        # Step 1: Rebalance the portfolio once
        date=datetime.now().date()
        liveportfolio.rebalance_live_portfolio(date, symbols)

    def get_live_metrics(self):
        while True:
            folder_path=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            updated_strategy_info=pd.read_json(os.path.join(folder_path, 'signal_generator/strategy_info.json'))
            db_manager = PortfolioDatabaseManager()
            liveportfolio = LivePortfolio(self.api, self.broker_config_path, updated_strategy_info,
                                          self.data_frequency, db_manager)
            current_datetime = time.strftime("%Y-%m-%d %H:%M")  # Fetch current timestamp
            liveportfolio._record_portfolio_state(current_datetime)  # Record portfolio state
            liveportfolio._query_portfolio_state()
            self.benchmark_returns=pd.DataFrame()
            close_prices=self.convert_portfolio_results(liveportfolio)
            portfolio_metrics=LiveMetrics(close_prices).compute_strategy_metrics(liveportfolio, self.benchmark_returns)
            portfolio_metrics['strategy_history']=pd.DataFrame(liveportfolio.strategy_history,
                                                               columns=['strategy_name', 'best_params', 'best_opti_algo'])
            return portfolio_metrics
