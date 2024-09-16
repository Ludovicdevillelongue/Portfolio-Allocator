from alloc_test.live.live_database_manager import PortfolioDatabaseManager


def is_estimation_period_satisfied(historical_returns, estimation_period):
    return len(historical_returns) >= estimation_period

def db_query_test():
    results=PortfolioDatabaseManager().query_portfolio_data()
    print(results)

if __name__ == "__main__":
    db_query_test()