from abc import ABC, abstractmethod

import gym
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import torch
from sklearn.ensemble import RandomForestRegressor
from stable_baselines3 import PPO, DDPG, TD3
from gym import spaces

class AllocationStrategy(ABC):
    @abstractmethod
    def compute_weights(self, historical_returns):
        pass


# Implementations of different allocation strategies
class OneOverN(AllocationStrategy):
    def compute_weights(self, historical_returns):
        num_assets = len(historical_returns.columns)
        return {symbol: 1.0 / num_assets for symbol in historical_returns.columns}


class MeanVar(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        mean_returns = historical_returns.mean().values
        cov_matrix = historical_returns.cov().values

        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_return = np.dot(weights, mean_returns)
            sharpe_ratio = (portfolio_return / np.sqrt(portfolio_variance)) * np.sqrt(252)
            return -sharpe_ratio  # maximize Sharpe Ratio

        num_assets = len(symbols)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets, ]

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(symbols, result.x))


class ERC(AllocationStrategy):
    def compute_weights(self, historical_returns):
        symbols = historical_returns.columns
        sigma = historical_returns.cov().values

        def objective(w):
            vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
            mri = (np.dot(sigma, w) / vol)
            rci = mri * w
            return np.sum((rci - rci.mean()) ** 2)

        result = minimize(
            objective,
            np.ones(len(symbols)) / len(symbols),
            method='SLSQP',
            bounds=[(0, 1)] * len(symbols),
            constraints=({'type': 'eq', 'fun': lambda x: x.sum() - 1.0}),
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
        transaction_costs = self.transaction_cost * np.sum(np.abs(action - self.current_weights)) * self.portfolio_value
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
        print(f"Step {self.current_step}: Portfolio Value = {self.portfolio_value}, Weights = {self.current_weights}")


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
        self.model.learn(total_timesteps=10000)  # Adjust timesteps based on the data length
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