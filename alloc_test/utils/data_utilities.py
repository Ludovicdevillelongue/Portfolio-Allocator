def is_estimation_period_satisfied(historical_returns, estimation_period):
    return len(historical_returns) >= estimation_period