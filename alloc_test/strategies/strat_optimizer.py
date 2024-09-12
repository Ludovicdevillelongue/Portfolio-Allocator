import itertools
import math
import os
import pandas as pd
import numpy as np
import random
import warnings
import pyfolio as pf
from abc import ABC, abstractmethod

class OptimizationAlgorithm(ABC):
    @abstractmethod
    def optimize(self, optimizer):
        pass

    def find_best_params(self, optimizer, evaluate):
        best_params = None
        best_sharpe_ratio = float('-inf')

        # Evaluate each parameter set
        for params in evaluate(optimizer):
            sharpe_ratio = optimizer.test_strategy(params)

            # Extract the optimization algorithm's class name and format it
            algo_name = self.__class__.__name__.replace('Algorithm', '')

            # Access strategy_instance from optimizer
            strat_test_recap = {
                'strat_name': optimizer.strategy_instance.__class__.__name__,  # Access strategy_instance from optimizer
                'opti_algo': algo_name,  # Properly formatted algorithm name (e.g., randomsearch, gridsearch, etc.)
                'params': params,
                'sharpe_ratio': sharpe_ratio
            }

            # Save the recap to the CSV
            strat_test_recap = pd.DataFrame([strat_test_recap])
            if not os.path.exists(optimizer.strat_opti_bt_csv) or os.path.getsize(optimizer.strat_opti_bt_csv) == 0:
                strat_test_recap.to_csv(optimizer.strat_opti_bt_csv, mode='w', header=True, index=False)
            else:
                strat_test_recap.to_csv(optimizer.strat_opti_bt_csv, mode='a', header=False, index=False)

            # Store the best Sharpe ratio and parameters
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                best_params = params

        return best_params, best_sharpe_ratio


class RandomSearchAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.random_evaluation)

    def random_evaluation(self, optimizer):
        for _ in range(optimizer.iterations):
            params = optimizer.generate_random_params()
            yield params

class GridSearchAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.grid_evaluation)

    def grid_evaluation(self, optimizer):
        param_values = []
        for key, value in optimizer.param_grids.items():
            if isinstance(value, tuple):
                min_val, max_val = value
                if isinstance(min_val, int):
                    param_values.append(range(min_val, max_val + 1))
                else:
                    param_values.append(np.arange(min_val, max_val, 0.1).tolist())
            elif isinstance(value, list):
                param_values.append(value)

        all_combinations = list(itertools.product(*param_values))  # Get all combinations first
        total_combinations = len(all_combinations)

        # Ensure the number of iterations does not exceed the total possible combinations
        max_iterations = min(optimizer.iterations, total_combinations)

        # Limit the combinations to the number of iterations provided
        limited_combinations = itertools.islice(all_combinations, max_iterations)

        for combination in limited_combinations:
            yield dict(zip(optimizer.param_grids.keys(), combination))

class SimulatedAnnealingAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.simulated_annealing_evaluation)

    def simulated_annealing_evaluation(self, optimizer):
        current_params = optimizer.generate_random_params()
        current_score = optimizer.test_strategy(current_params)
        temp = 1.0
        cooling_rate = 0.9

        for _ in range(optimizer.iterations):
            new_params = optimizer.generate_random_params()
            new_score = optimizer.test_strategy(new_params)

            exponent = (current_score - new_score) / temp
            exponent = max(exponent, -700)
            if new_score > current_score or math.exp(exponent) > random.random():
                current_score = new_score
                current_params = new_params
                yield current_params

            temp *= cooling_rate

class GeneticAlgorithm(OptimizationAlgorithm):
    def optimize(self, optimizer):
        return self.find_best_params(optimizer, self.genetic_algorithm_evaluation)

    def genetic_algorithm_evaluation(self, optimizer):
        population_size = optimizer.iterations
        mutation_rate = 0.1
        population = [optimizer.generate_random_params() for _ in range(population_size)]
        scores = [optimizer.test_strategy(params) for params in population]

        for _ in range(optimizer.iterations):
            new_population = self.evolve_population(population, scores, mutation_rate)
            scores = [optimizer.test_strategy(params) for params in new_population]
            population = new_population
            best_index = np.argmax(scores)
            yield population[best_index]

    def evolve_population(self, population, scores, mutation_rate):
        new_population = []
        for _ in range(len(population) // 2):
            parent1, parent2 = self.select_parents(population, scores)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.extend([
                self.mutate(child1, mutation_rate),
                self.mutate(child2, mutation_rate)
            ])
        return new_population

    def select_parents(self, population, scores):
        probabilities = np.exp(scores) / np.sum(np.exp(scores))
        parent_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
        return population[parent_indices[0]], population[parent_indices[1]]

    def crossover(self, parent1, parent2):
        if len(parent1) > 1:
            crossover_point = random.randint(1, len(parent1) - 1)
        else:
            # Handle the case where parent1 is too short for crossover
            crossover_point = 0
        child1 = {**dict(list(parent1.items())[:crossover_point]),
                  **dict(list(parent2.items())[crossover_point:])}
        child2 = {**dict(list(parent2.items())[:crossover_point]),
                  **dict(list(parent1.items())[crossover_point:])}
        return child1, child2

    def mutate(self, individual, mutation_rate):
        for key in individual:
            if random.random() < mutation_rate:
                if isinstance(individual[key], int):
                    individual[key] += random.randint(-1, 1)
                elif isinstance(individual[key], float):
                    individual[key] += random.uniform(-0.1, 0.1)
        return individual


class StrategyOptimizer:
    def __init__(self, strategy_instance, param_grids, optimization_algorithms, iterations, strategy_runner, strat_opti_bt_csv):
        self.strategy_instance = strategy_instance
        self.param_grids = param_grids
        self.optimization_algorithms = optimization_algorithms
        self.iterations = iterations
        self.strategy_runner = strategy_runner
        self.strat_opti_bt_csv = strat_opti_bt_csv
        self.global_best_sharpe = float('-inf')
        self.global_best_params = None
        self.global_best_opti_algo = None

    def generate_random_params(self):
        """Generate a random set of parameters from the parameter grid."""
        params = {}
        for key, param_range in self.param_grids.items():
            if isinstance(param_range, tuple):
                min_val, max_val = param_range
                if isinstance(min_val, int):
                    params[key] = random.randint(min_val, max_val)
                else:
                    params[key] = random.uniform(min_val, max_val)
            elif isinstance(param_range, list):
                params[key] = random.choice(param_range)
        return params

    def test_strategy(self, strategy_params):
        """Test the strategy using the provided parameters."""
        # Set the parameters on the existing strategy instance
        for param_name, param_value in strategy_params.items():
            setattr(self.strategy_instance, param_name, param_value)

        # Run the strategy and calculate the Sharpe ratio
        portfolio_metrics = self.strategy_runner.run_allocation(self.strategy_instance)
        sharpe_ratio = pf.timeseries.sharpe_ratio(portfolio_metrics['portfolio_returns'])
        return sharpe_ratio

    def _optimize_strategy(self, optimization_algorithm):
        best_params, best_sharpe = optimization_algorithm.optimize(self)
        if best_sharpe > self.global_best_sharpe:
            self.global_best_sharpe = best_sharpe
            self.global_best_params = best_params
            self.global_best_opti_algo = optimization_algorithm.__class__.__name__

    def test_all_search_types(self):
        best_results = {}
        for search_type in self.optimization_algorithms:
            self._optimize_strategy(search_type)
        return self.global_best_opti_algo, self.global_best_params