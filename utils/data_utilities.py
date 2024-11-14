import pandas as pd

from live.live_database_manager import PortfolioDatabaseManager


def is_in_out_sample_period_satisfied(historical_returns, in_out_sample_period):
    return len(historical_returns) >= in_out_sample_period

def db_query_test():
    (position_history, price_history, weight_history,transaction_history, cash_history,strategy_history)\
        =PortfolioDatabaseManager().query_portfolio_data()
    position_history=pd.DataFrame(position_history)
    price_history=pd.DataFrame(price_history)
    weight_history=pd.DataFrame(weight_history)
    transaction_history=pd.DataFrame(transaction_history)
    cash_history=pd.DataFrame(cash_history)
    strategy_history=pd.DataFrame(strategy_history)
    return 0




def db_delete():
    PortfolioDatabaseManager().delete_tables()
    print("Tables deleted")

if __name__ == "__main__":
    db_query_test()