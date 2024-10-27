from live.live_database_manager import PortfolioDatabaseManager


def is_in_out_sample_period_satisfied(historical_returns, in_out_sample_period):
    return len(historical_returns) >= in_out_sample_period

def db_query_test():
    results=PortfolioDatabaseManager().query_portfolio_data()
    print(results)

def db_delete():
    PortfolioDatabaseManager().delete_tables()
    print("Tables deleted")

if __name__ == "__main__":
    db_delete()