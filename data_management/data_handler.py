class DataHandler:
    def adjust_prices_for_splits(self, prices, stock_split_ratios):
        adjusted_prices = prices.copy()
        for symbol in stock_split_ratios.columns:
            for date in stock_split_ratios.index:
                if stock_split_ratios.loc[date, symbol] != 1.0:  # If there is a split
                    split_ratio = stock_split_ratios.loc[date, symbol]
                    # Adjust all prices before the split
                    adjusted_prices.loc[adjusted_prices.index < date, symbol] /= split_ratio
        return adjusted_prices

    def adjust_volumes_for_splits(self, volumes, stock_split_ratios):
        adjusted_volumes = volumes.copy()
        for symbol in stock_split_ratios.columns:
            for date in stock_split_ratios.index:
                if stock_split_ratios.loc[date, symbol] != 1.0:  # If there is a split
                    split_ratio = stock_split_ratios.loc[date, symbol]
                    # Adjust all volumes before the split
                    adjusted_volumes.loc[adjusted_volumes.index < date, symbol] *= split_ratio
        return adjusted_volumes