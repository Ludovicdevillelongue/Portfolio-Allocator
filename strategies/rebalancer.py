from datetime import timedelta, datetime


class Rebalancer:
    def __init__(self, allocation_strategy, rebalance_frequency, last_rebalance_date=None):
        self.allocation_strategy = allocation_strategy
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance_date = (datetime.strptime(last_rebalance_date, '%Y-%m-%d').date()
                                    if isinstance(last_rebalance_date, str)
                                    else last_rebalance_date)

    def should_rebalance(self, current_date):
        if self.last_rebalance_date is None:
            return True

        if self.rebalance_frequency == 'daily':
            return current_date != self.last_rebalance_date
        elif self.rebalance_frequency == 'weekly':
            return (current_date - self.last_rebalance_date) >= timedelta(days=7)
        elif self.rebalance_frequency == 'monthly':
            return (current_date.year, current_date.month) != (self.last_rebalance_date.year, self.last_rebalance_date.month)
        else:
            raise ValueError(f"Unsupported rebalance frequency: {self.rebalance_frequency}")

    def rebalance(self, current_date, historical_returns):
        if self.should_rebalance(current_date):
            weights = self.allocation_strategy.compute_weights(historical_returns)
            self.last_rebalance_date = current_date
            return weights
        return None