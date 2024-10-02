class BenchmarkPortfolio:
    def __init__(self, symbols, volumes):
        self.symbols = symbols
        self.volumes = volumes
        self.volume_weights = self._calculate_volume_weights()

    def _calculate_volume_weights(self):
        total_volumes = self.volumes.sum(axis=1)
        volume_weights = self.volumes.div(total_volumes, axis=0)
        return volume_weights

    def compute_benchmark_returns(self, asset_returns):
        aligned_weights = self.volume_weights.loc[asset_returns.index]
        weighted_returns = asset_returns * aligned_weights
        benchmark_returns = weighted_returns.sum(axis=1)
        return benchmark_returns
