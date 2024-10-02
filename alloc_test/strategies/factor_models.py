from sklearn.linear_model import LinearRegression

class FactorModel:
    def __init__(self, factors):
        self.factors = factors
        self.model = LinearRegression()

    def fit(self, returns):
        aligned_data = returns.align(self.factors, join='inner')
        self.model.fit(aligned_data[1], aligned_data[0])

    def get_factor_loadings(self):
        return dict(zip(self.factors.columns, self.model.coef_))

    def get_alpha(self):
        return self.model.intercept_

    def calculate_factor_returns(self, returns):
        factor_returns = {}

        for symbol in returns.columns:
            self.fit(returns[symbol])
            factor_returns[symbol] = {
                'alpha': self.get_alpha(),
                'factor_loadings': self.get_factor_loadings()
            }

        return factor_returns