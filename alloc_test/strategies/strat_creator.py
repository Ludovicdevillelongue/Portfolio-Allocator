import numpy as np
from abc import ABC, abstractmethod
import gym
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import PPO, DDPG, TD3
from gym import spaces

class AllocationStrategy(ABC):
    @abstractmethod
    def compute_weights(self, historical_returns):
        pass

class ERC(AllocationStrategy):
    def __init__(self, regularization_strength=0.0):
        self.regularization_strength = regularization_strength

    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            mri = np.dot(sigma, w) / vol
            rci = mri * w
            return np.sum((rci - rci.mean()) ** 2) + self.regularization_strength * np.sum(w ** 2)

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))


# Implementations of different allocation strategies
class OneOverN(AllocationStrategy):
    def compute_weights(self, historical_returns):
        num_assets = len(historical_returns.columns)
        return {symbol: 1.0 / num_assets for symbol in historical_returns.columns}

class MeanVar(AllocationStrategy):
    def __init__(self, risk_free_rate=0.0, target_return=None):
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        mean_returns = historical_returns.mean().values
        cov_matrix = historical_returns.cov().values
        def objective(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            sharpe_ratio = ((portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)) * np.sqrt(252)
            return -sharpe_ratio  # Negative because we want to maximize it

        num_assets = len(symbols)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights must sum to 1

        if self.target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - self.target_return})

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=constraints,
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))




class MinVol(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            return np.sqrt(np.dot(np.dot(w, sigma), w))

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))

class HierarchicalRiskParity(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        corr_matrix = historical_returns.corr().values
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        try:
            link = linkage(squareform(dist_matrix), 'single')
            clusters = fcluster(link, 0.5, criterion='distance')

            weights = np.zeros(len(symbols))
            cluster_allocations = {i: 1 / len(set(clusters)) for i in set(clusters)}

            for i, symbol in enumerate(symbols):
                weights[i] = cluster_allocations[clusters[i]]

            weights /= np.sum(weights)
        except Exception as e:
            weights = np.zeros(len(symbols))
        return dict(zip(symbols, weights))


class AdvancedHierarchicalRiskParity(AllocationStrategy):
    class HierarchicalRiskParity(AllocationStrategy):
        def __init__(self,
                     linkage_method='single',
                     distance_metric='euclidean',
                     risk_measure='variance',
                     allocation_method='inverse_variance',
                     neutralize_beta=False,
                     bounds=(0, 1)):
            self.linkage_method = linkage_method
            self.distance_metric = distance_metric
            self.risk_measure = risk_measure
            self.allocation_method = allocation_method
            self.neutralize_beta = neutralize_beta
            self.bounds = bounds

        def compute_weights(self, historical_returns):
            symbols = historical_returns.columns
            # 1. Compute the covariance and correlation matrices
            cov = historical_returns.cov()
            corr = historical_returns.corr()

            # 2. Compute the distance matrix using the selected distance metric
            if self.distance_metric == 'euclidean':
                dist = np.sqrt((1 - corr) / 2)
            else:
                dist = pd.DataFrame(
                    squareform(pdist(corr, metric=self.distance_metric)),
                    index=corr.index,
                    columns=corr.index
                )

            dist_condensed = squareform(dist.values)

            # 3. Perform hierarchical clustering using the selected linkage method
            linkage_matrix = linkage(dist_condensed, method=self.linkage_method)

            # 4. Get the order of assets from clustering
            sorted_indices = self._get_quasi_diag(linkage_matrix)
            sorted_symbols = historical_returns.columns[sorted_indices]

            # 5. Arrange covariance matrix accordingly
            ordered_cov = cov.loc[sorted_symbols, sorted_symbols]

            # 6. Compute hierarchical risk parity weights
            weights = self._get_recursive_bisection(ordered_cov)

            # 7. Reindex weights to original asset order and apply bounds
            hrp_weights = weights.reindex(historical_returns.columns).fillna(0)
            hrp_weights = hrp_weights.clip(self.bounds[0], self.bounds[1])

            # 8. Normalize weights to sum to 1
            hrp_weights /= hrp_weights.sum()

            # 9. Neutralize market beta if required
            if self.neutralize_beta:
                betas = historical_returns.apply(
                    lambda x: x.cov(historical_returns.mean()) / historical_returns.mean().var())
                portfolio_beta = np.dot(hrp_weights, betas)
                hrp_weights -= portfolio_beta * betas / betas.sum()
                hrp_weights = hrp_weights.clip(self.bounds[0], self.bounds[1])
                hrp_weights /= hrp_weights.sum()

            return dict(zip(symbols, hrp_weights))

        def _get_quasi_diag(self, linkage_matrix):
            n = linkage_matrix.shape[0] + 1
            index = self._quasi_diag(linkage_matrix, n)
            return index

        def _quasi_diag(self, linkage_matrix, n):
            # Sorts clusters recursively to build the ordered list of assets
            def recursive_sort(cluster_id):
                if cluster_id < n:
                    return [cluster_id]
                else:
                    left = int(linkage_matrix[cluster_id - n, 0])
                    right = int(linkage_matrix[cluster_id - n, 1])
                    return recursive_sort(left) + recursive_sort(right)

            return recursive_sort(2 * n - 2)

        def _get_recursive_bisection(self, cov):
            weights = pd.Series(1, index=cov.index)
            clusters = [cov.index.tolist()]
            while clusters:
                clusters_next = []
                for cluster in clusters:
                    if len(cluster) > 1:
                        # Split the cluster into two
                        split = int(len(cluster) / 2)
                        cluster_left = cluster[:split]
                        cluster_right = cluster[split:]

                        # Compute cluster risks
                        risk_left = self._get_cluster_risk(cov, cluster_left)
                        risk_right = self._get_cluster_risk(cov, cluster_right)

                        # Allocate weights based on selected risk measure
                        alloc_factor = risk_right / (risk_left + risk_right)
                        weights[cluster_left] *= alloc_factor
                        weights[cluster_right] *= (1 - alloc_factor)

                        # Add sub-clusters to the list
                        clusters_next.extend([cluster_left, cluster_right])
                clusters = clusters_next
            return weights

        def _get_cluster_risk(self, cov, cluster):
            cov_slice = cov.loc[cluster, cluster]
            if self.allocation_method == 'inverse_variance':
                inv_var = 1 / np.diag(cov_slice)
                weights = inv_var / inv_var.sum()
            else:
                weights = np.ones(len(cov_slice)) / len(cov_slice)
            portfolio_variance = np.dot(weights.T, np.dot(cov_slice.values, weights))
            if self.risk_measure == 'variance':
                return portfolio_variance
            elif self.risk_measure == 'standard_deviation':
                return np.sqrt(portfolio_variance)
            # Implement other risk measures if needed
            else:
                raise ValueError(f"Unknown risk measure: {self.risk_measure}")

class MaximumDivergence(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values
        std_dev = np.sqrt(np.diag(sigma))

        def objective(w):
            std_dev_portfolio = np.sqrt(np.dot(np.dot(w, sigma), w))
            weighted_std_dev = np.dot(w, std_dev)
            divergence = weighted_std_dev / std_dev_portfolio
            return -1. * np.log(divergence.sum())

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
            options={'disp': False, 'ftol': 1e-10})

        return dict(zip(symbols, result.x))

class MLModelAllocator(AllocationStrategy):
    def __init__(self, model=None):
        if model is None:
            self.model = RandomForestRegressor()
        else:
            self.model = model

    def compute_weights(self, historical_returns):
        # Train ML model to predict future returns
        features = historical_returns.dropna()
        target = historical_returns.shift(-1).dropna()  # Predict next period returns
        self.model.fit(features, target)

        # Predict next-period returns and compute weights
        next_period_returns = self.model.predict(features.iloc[-1:])
        weights = self._optimize_weights(next_period_returns)
        return weights

    def _optimize_weights(self, predicted_returns):
        num_assets = len(predicted_returns)
        weights = np.ones(num_assets) / num_assets
        return {symbol: weight for symbol, weight in zip(predicted_returns.columns, weights)}

class PortfolioEnv(gym.Env):
    def __init__(self, historical_returns, initial_capital=100000, transaction_cost=0.001, risk_aversion=1.0):
        super(PortfolioEnv, self).__init__()
        self.historical_returns = historical_returns
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion

        self.num_assets = historical_returns.shape[1]
        self.current_step = 0
        self.done = False

        # Portfolio state: asset weights, cash position, market conditions (returns)
        self.state_size = self.num_assets + 2  # Weights + Cash + Returns info
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)

        self.current_weights = np.zeros(self.num_assets)
        self.cash_balance = self.initial_capital
        self.portfolio_value = self.initial_capital

    def reset(self):
        self.current_step = 0
        self.done = False
        self.current_weights = np.ones(self.num_assets) / self.num_assets  # Equal weight start
        self.cash_balance = self.initial_capital
        self.portfolio_value = self.initial_capital
        return self._get_observation()

    def _get_observation(self):
        current_returns = self.historical_returns.iloc[self.current_step].values
        return np.concatenate([self.current_weights, [self.cash_balance], [current_returns.mean()]])

    def step(self, action):
        if self.done:
            raise ValueError("Cannot step in a finished episode")

        action = np.clip(action, 0, 1)  # Ensure valid action (weights between 0 and 1)
        action /= np.sum(action)  # Normalize weights to sum to 1

        # Market returns and portfolio adjustment
        current_returns = self.historical_returns.iloc[self.current_step].values
        self.portfolio_value = (self.portfolio_value * np.dot(self.current_weights, 1 + current_returns))

        # Transaction costs for rebalancing
        transaction_costs = self.transaction_cost * np.sum(
            np.abs(action - self.current_weights)) * self.portfolio_value
        self.portfolio_value -= transaction_costs

        # Update weights and cash balance
        self.current_weights = action
        self.cash_balance = self.portfolio_value

        # Calculate reward (e.g., risk-adjusted return using Sharpe ratio)
        reward = (self.portfolio_value - self.initial_capital) / self.initial_capital
        reward -= self.risk_aversion * np.std(current_returns)  # Penalize high volatility

        self.current_step += 1
        if self.current_step >= len(self.historical_returns) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self, mode='human'):
        print(
            f"Step {self.current_step}: Portfolio Value = {self.portfolio_value}, Weights = {self.current_weights}")

# Reinforcement Learning Allocator using PPO or TD3
class ReinforcementLearningAllocator(AllocationStrategy):
    def __init__(self, historical_returns, algorithm='PPO', transaction_cost=0.001, risk_aversion=1.0):
        self.env = PortfolioEnv(historical_returns, transaction_cost=transaction_cost, risk_aversion=risk_aversion)
        self.algorithm = algorithm
        self.model = self._select_rl_model()

    def _select_rl_model(self):
        if self.algorithm == 'PPO':
            return PPO("MlpPolicy", self.env, verbose=1)
        elif self.algorithm == 'DDPG':
            return DDPG("MlpPolicy", self.env, verbose=1)
        elif self.algorithm == 'TD3':
            return TD3("MlpPolicy", self.env, verbose=1)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _train_agent(self, historical_returns):
        # Training the agent with the environment
        self.model.learn(total_timesteps=10000)  # Adjust timesteps based on the data_management length
        return self.model

    def compute_weights(self, historical_returns):
        agent = self._train_agent(historical_returns)
        obs = self.env.reset()
        done = False
        while not done:
            action, _states = agent.predict(obs)
            obs, rewards, done, info = self.env.step(action)
        return dict(zip(historical_returns.columns, self.env.current_weights))

class CVaROptimization(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        cov_matrix = historical_returns.cov().values

        def objective(weights):
            portfolio_returns = np.dot(weights, historical_returns.T)
            VaR = np.percentile(portfolio_returns, 5)
            CVaR = np.mean(portfolio_returns[portfolio_returns < VaR])
            return -CVaR  # Minimize Conditional VaR

        num_assets = len(symbols)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(symbols, result.x))

class EfficientFrontier(AllocationStrategy):
    def __init__(self, target_return=0.1):
        self.target_return = target_return

    def compute_weights(self, historical_returns):
        cov_matrix = historical_returns.cov()
        mean_returns = historical_returns.mean()
        num_assets = len(mean_returns)
        bounds = [(0, 1) for _ in range(num_assets)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - self.target_return}]
        result = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
                          np.ones(num_assets) / num_assets, bounds=bounds, constraints=constraints)
        return dict(zip(historical_returns.columns, result.x))