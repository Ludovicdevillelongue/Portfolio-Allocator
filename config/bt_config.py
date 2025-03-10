

# Backtest configuration
from datetime import datetime, timedelta

# General settings
data_frequency = 'day'
broker = 'alpaca'
symbols = [
    'TSLA', 'NVDA', 'MSFT'
]
start_date = (datetime.now() - timedelta(days=4 * 365)).strftime('%Y-%m-%d')
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
initial_capital = 100000
rebalance_frequency = 'daily'

# Backtest settings
in_out_sample_period = 90
iterations = 10
bt_port = 6001


# Strategies to use
from strategies.strat_creator import (MinVol,MaximumDivergence, MeanVar, ERC, CVaROptimization, HierarchicalRiskParity,
                                  AdvancedHierarchicalRiskParity, MLModelAllocator, ReinforcementLearningAllocator)
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

strategies = {
    'MinVol': MinVol(),
    'MaximumDivergence': MaximumDivergence(),
    'MeanVar': MeanVar(),
    'ERC': ERC(),
    'CVaROptimization': CVaROptimization(),
    'HierarchicalRiskParity': HierarchicalRiskParity(),
    'MLModelAllocator': MLModelAllocator(),
    'ReinforcementLearningAllocator': ReinforcementLearningAllocator(),
    # 'AdvancedHierarchicalRiskParity': AdvancedHierarchicalRiskParity(),
}

# Parameter grids for strategy optimization
param_grids = {
    'MinVol': {
        'regularization_strength': (0.0, 0.1)
    },
    'MaximumDivergence': {
        'regularization_strength': (0.0, 0.1)
    },
    'MeanVar': {
        'risk_free_rate': (0.01, 0.05),
        'target_return': (0.05, 0.15)
    },
    'ERC': {
        'regularization_strength': (0.0, 0.1)
    },
    'CVaROptimization': {
        'confidence_level': (0.9, 0.99)
    },
    'HierarchicalRiskParity': {
        'linkage_method': ['single', 'complete', 'average', 'ward'],
        'distance_metric': ['correlation', 'cosine', 'dice', 'euclidean', 'hamming']
    },

    'AdvancedHierarchicalRiskParity': {
        'linkage_method': ['single', 'complete', 'average', 'ward'],
        'distance_metric': ['correlation', 'cosine', 'dice', 'euclidean', 'hamming'],
        'risk_measure': ['variance', 'standard_deviation'],
        'allocation_method': ['inverse_variance', 'equal_weight'],
        'neutralize_beta': [True, False],
        'bounds': [(0, 1)]
    },
    'MLModelAllocator': {
        'model': [RandomForestRegressor(), XGBRegressor()],
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'regularization_strength': (0.0, 0.1)
    },
    'ReinforcementLearningAllocator': {
        'algorithm': ['PPO', 'TD3', 'DDPG'],
        'risk_aversion': (0.5, 2.0),
        'total_timesteps': (100, 500)
    }
}

# Optimization algorithms
from strategies.strat_optimizer import RandomSearchAlgorithm, GridSearchAlgorithm
optimization_algorithms = [RandomSearchAlgorithm(), GridSearchAlgorithm()]


# File paths
import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File paths
broker_config_path = os.path.join(project_root, 'config', 'broker_config.yml')
strat_opti_bt_csv = os.path.join(project_root, 'strategies', 'alloc_strat_opti_tester_recap.csv')
strategy_info_path = os.path.join(project_root, 'signal_generator', 'strategy_info.json')