import os
import warnings

warnings.filterwarnings('ignore')
import alpaca_trade_api as tradeapi
import pandas as pd
import yaml
class TradingPlatform:
    """Base class for all trading platforms."""

    def api_connection(self):
        raise NotImplementedError

    def get_account_info(self):
        raise NotImplementedError

    def get_all_orders(self):
        raise NotImplementedError

    def get_all_positions(self):
        raise NotImplementedError

    def get_broker_portfolio_history(self):
        raise NotImplementedError

    def get_portfolio_metrics(self, risk_free_rate):
        raise NotImplementedError


class AlpacaPlatformMetrics(TradingPlatform):
    """Alpaca trading platform implementation."""

    def __init__(self, api, data_frequency):
        self.api=api
        self.data_frequency = data_frequency
        self.equity_value_tracker_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                     f'results\\{self.data_frequency}_broker_equity_value.csv')
        self.pos_returns_tracker_yml = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                    f'results\\{self.data_frequency}_broker_pos_ret.yml')
        self.ptf_metrics_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                            f'results\\{self.data_frequency}_broker_ptf_metrics.csv')


    def get_account_info(self):
        account = self.api.get_account()
        dict_account_info = {
            'currency': account.currency,
            'pending_transfer_in': account.pending_transfer_in,
            'created_at': account.created_at,
            'position_market_value': account.position_market_value,
            'cash': account.cash,
            'accrued_fees': account.accrued_fees,
            'buying_power': account.buying_power,
            'portfolio_value': account.portfolio_value
        }
        df_account_info = pd.DataFrame.from_dict(dict_account_info, orient='index').T
        return df_account_info

    def get_portfolio_value(self):
        account = self.api.get_account()
        return float(account.equity)

    def get_all_orders(self):
        orders = self.api.list_orders(status='all')
        orders_list = [order._raw for order in orders]
        df_orders = pd.DataFrame(orders_list)
        if df_orders.empty:
            return pd.DataFrame()
        else:
            df_orders['created_at'] = pd.to_datetime(df_orders['created_at']).dt.tz_convert(
                'Europe/Paris')
            df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at']).dt.tz_convert(
                'Europe/Paris')
            return df_orders[['created_at', 'filled_at', 'asset_id', 'symbol',
                              'asset_class', 'qty', 'filled_qty', 'order_type', 'side', 'filled_avg_price',
                              'time_in_force', 'limit_price', 'stop_price']]

    def get_symbol_orders(self, symbol):
        orders = self.api.list_orders(status='all', symbols=[symbol])
        orders_list = [order._raw for order in orders]
        df_orders = pd.DataFrame(orders_list)

        if df_orders.empty:
            return pd.DataFrame()
        else:
            df_orders['created_at'] = pd.to_datetime(df_orders['created_at']).dt.tz_convert(
                'Europe/Paris')
            df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at']).dt.tz_convert(
                'Europe/Paris')
            return df_orders[['created_at', 'filled_at', 'asset_id', 'symbol',
                              'asset_class', 'qty', 'filled_qty', 'order_type', 'side', 'filled_avg_price',
                              'time_in_force', 'limit_price', 'stop_price']]

    def get_all_positions(self):
        positions = self.api.list_positions()
        positions_list = [position._raw for position in positions]
        df_positions = pd.DataFrame(positions_list)
        return df_positions

    def get_symbol_position(self, symbol):
        symbol = symbol.replace("/", "") if '/' in symbol else symbol
        position = self.api.get_position(symbol)
        pos = pd.DataFrame(pd.Series(position._raw)).T
        pos = pos.round(2)
        return pos
    def get_assets(self):
        assets = self.api.list_assets()
        return assets

    def create_positions_pnl_table(self):
        df_positions = self.get_all_positions()
        try:
            df_positions = df_positions[
                ['symbol', 'current_price', 'qty', 'side', 'market_value', 'unrealized_pl']].round(2)
        except Exception as e:
            df_positions = pd.DataFrame()
        return df_positions

    def create_orders_table(self):
        df_orders = self.get_all_orders()
        df_orders = df_orders[df_orders['filled_at'].notna()]
        df_orders['filled_at'] = pd.to_datetime(df_orders['filled_at'])
        df_orders = df_orders.sort_values(by='filled_at')
        df_orders = df_orders.reset_index(drop=True)

        # Calculate PnL for each order
        df_orders['pnl'] = 0.0
        symbols = df_orders['symbol'].unique()
        all_orders_list=[]
        for symbol in symbols:
            symbol_orders = df_orders[df_orders['symbol'] == symbol]
            symbol_orders = symbol_orders.reset_index(drop=True)
            for i in range(1, len(symbol_orders)):
                previous_order = symbol_orders.iloc[i - 1]
                current_order = symbol_orders.iloc[i]
                price_diff = float(current_order['filled_avg_price']) - float(previous_order['filled_avg_price'])
                qty = float(previous_order['filled_qty'])
                if previous_order['side'] == 'buy' and current_order['side'] == 'sell':
                    symbol_orders.at[i - 1, 'pnl'] = price_diff * qty
                elif previous_order['side'] == 'sell' and current_order['side'] == 'buy':
                    symbol_orders.at[i - 1, 'pnl'] = -price_diff * qty
            all_orders_list.append(symbol_orders)
        all_orders=pd.concat(all_orders_list)
        all_orders = all_orders.sort_values('filled_at', ascending=False).round(2)
        return all_orders

    def get_broker_portfolio_history(self):
        portfolio_history = self.api.get_portfolio_history(period='1W', timeframe='1Min', extended_hours=True).df
        return portfolio_history

    def get_all_portfolio_history(self):
        df_ptf_last_day = self.get_broker_portfolio_history()
        df_ptf_last_day = df_ptf_last_day.tz_convert('Europe/Paris')
        try:
            df_ptf_history = pd.read_csv(self.equity_value_tracker_csv, header=[0], index_col=[0])
            df_ptf_history.index = pd.to_datetime(df_ptf_history.index).tz_convert('Europe/Paris')
            df_ptf = df_ptf_last_day.combine_first(df_ptf_history)
        except Exception as e:
            df_ptf = df_ptf_last_day
        df_ptf = df_ptf[df_ptf['equity'] > 100]
        df_ptf.to_csv(self.equity_value_tracker_csv, mode='w', header=True, index=True)
        return df_ptf
