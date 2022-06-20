import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

from torch import initial_seed

import strategies, backtester, statistics


def sort_population(population, scores):
    """Sort the population based on its scores (from highest to lowest score)"""
    df = pd.DataFrame()
    df['index'] = [idx for idx in range(len(population))]
    df['score'] = scores
    df.sort_values(by="score", ascending=False, inplace=True)
    sorted_index = df['index'].values.tolist()
    population = np.array(population)[sorted_index]
    population = population.tolist()
    scores = np.array(scores)[sorted_index]
    scores = scores.tolist()
    return population, scores


def selection(population, size):
    """
    Select the elements showing the highest scores.
    size is the number of elements to select.
    """
    selected_population = population[:size]
    return selected_population


def crossover(p1, p2, r_cross):
    """Create two new children out of the parents p1 and p2."""
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() < r_cross:
        # choose a random point where the crossover must be done
        pt = np.random.randint(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def mutation(genes, r_mut):
    """Apply mutation on the given genom with an r_mut probability."""
    for i in range(len(genes)):
        if np.random.rand() < r_mut:
            genes[i] = np.random.rand()
    return genes


def genetic_search(objective, n_bits, n_iter, n_pop, n_parents, n_bests, r_cross, r_mut):
    """Maximize the objective function through a genetic search."""
    # Initialize population
    population = [np.random.rand(n_bits).tolist() for _ in range(n_pop)]
    # Keep track of the best solution
    best, best_eval = 0, objective(population[0])
    # Run generations
    for g in range(n_iter):
        # Evaluate all candidates in the population
        scores = [objective(c) for c in population]
        # Sort population according to the scores
        population, scores = sort_population(population, scores)
        # Check the new best solution
        if scores[0] > best_eval:
            best, best_eval = population[0], scores[0]
            print(f">New best: f({population[0]}) = {scores[0]}")
        # Select parents
        selected = selection(population, n_parents)
        # Create the next generation
        children = list()
        # Keep the n_bests parents as they are
        for i in range(n_bests):
            children.append(n_bests)
        # Create new children
        while len(children) < len(population):
            i1, i2 = np.random.randint(0, len(selected)), np.random.randint(0, len(selected))
            p1, p2 = selected[i1], selected[i2]
            for c in crossover(p1, p2, r_cross):
                if len(children) < len(population):
                    c = mutation(c, r_mut)
                    children.append(c)
        population = children
    return [best, best_eval]




class GeneticOptimizer(object):
    
    def __init__(self, objective, constraint, bounds, fitness_settings, n_iter, n_pop, n_parents, n_bests, r_cross, r_mut):
        self.objective = objective
        self.constraint = constraint if constraint != None else lambda x: True
        self.dimension = len(bounds)
        self.fitness_settings = fitness_settings
        self.bounds = bounds
        self.types = [typ for lb, ub, typ in bounds]
        self.lower_bounds = [lb for lb, ub, typ in bounds]
        self.upper_bounds = [ub for lb, ub, typ in bounds]
        self.space_widths = [ub - lb for lb, ub, typ in bounds]
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.n_parents = n_parents
        self.n_bests = n_bests
        self.r_cross = r_cross
        self.r_mut = r_mut
        return

    def binary(self):
        """Return True if all genes are binaries (integers between 0 and 1), else False"""
        condition1 = all([lb == 0 for lb in self.lower_bounds])
        condition2 = all([ub == 1 for ub in self.upper_bounds])
        condition3 = all([typ == int for typ in self.types])
        return (condition1 and condition2 and condition3)

    def new_genom(self):
        """Create a new random genome respecting the bounds and types"""
        genom = []
        for i in range(self.dimension):
            if self.binary():
                gene = np.random.randint(0,2)
            else:
                gene = np.random.rand()
            genom.append(gene)
        if self.constraint(self.denormalize(genom)):
            return genom
        return self.new_genom()

    def normalize(self, degenom: list):
        """From a degenom of values within bounds, return the corresponding genom values between 0 and 1."""
        if self.binary():
            genom = degenom
        else:
            genom = [(degenom[i] - self.lower_bounds[i]) / self.space_widths[i] for i in range(self.dimension)]
        return genom

    def denormalize(self, genom: list):
        """From a genom of values between 0 and 1, return the corresponding parameters within the bounds."""
        if self.binary():
            degenom = genom
        else:
            degenom = [self.types[i](genom[i] * self.space_widths[i] + self.lower_bounds[i]) for i in range(self.dimension)]
        return degenom

    def sort_population(self, population: list, scores: list):
        """Sort the population based on its scores (from highest to lowest score)"""
        df = pd.DataFrame()
        df['index'] = [idx for idx in range(len(population))]
        df['score'] = scores
        df.sort_values(by="score", ascending=False, inplace=True)
        sorted_index = df['index'].values.tolist()
        population = np.array(population)[sorted_index]
        population = population.tolist()
        scores = np.array(scores)[sorted_index]
        scores = scores.tolist()
        return population, scores

    def selection(self, sorted_population: list, size: int):
        """
        Select the elements showing the highest scores.
        size is the number of elements to select.
        """
        selected_population = sorted_population[:size]
        return selected_population

    def crossover(self, p1: list, p2: list, r_cross: float):
        """Create two new children out of the parents p1 and p2."""
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross:
            # choose a random point where the crossover must be done
            pt = np.random.randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def mutation(self, genes: list, r_mut: float):
        """Apply mutation on the given genom with an r_mut probability."""
        for i in range(len(genes)):
            if np.random.rand() < r_mut:
                if self.binary():
                    genes[i] = np.abs(genes[i] - 1)
                else:
                    genes[i] = np.random.rand()
        return genes

    def search(self, initial_population=None):
        """Maximize the objective function through a genetic search."""
        # Initialize population
        population = initial_population if initial_population != None else []
        while len(population) < self.n_pop:
            population.append(self.new_genom())
        # Keep track of the best solution
        best, best_eval = 0, self.objective(self.denormalize(population[0]), self.bounds, self.fitness_settings)
        # Run generations
        for g in range(n_iter):
            # Evaluate all candidates in the population
            scores = [self.objective(self.denormalize(c), self.bounds, self.fitness_settings) for c in population]
            # Sort population according to the scores
            population, scores = self.sort_population(population, scores)
            # Check the new best solution
            if scores[0] > best_eval:
                best, best_eval = population[0], scores[0]
                print(f">New best: f({self.denormalize(population[0])}) = {scores[0]}")
            # Select parents
            selected = self.selection(population, self.n_parents)
            # Create the next generation
            children = list()
            # Keep the n_bests parents as they are
            for i in range(self.n_bests):
                children.append(selected[i])
            # Create new children
            while len(children) < len(population):
                i1, i2 = np.random.randint(0, len(selected)), np.random.randint(0, len(selected))
                p1, p2 = selected[i1], selected[i2]
                for c in self.crossover(p1, p2, r_cross):
                    if len(children) < len(population):
                        c = self.mutation(c, r_mut)
                        children.append(c)
            population = children
        return [best, best_eval]




def is_valid(degenom: list):
    """Verify that the given degenom is a valid genom."""
    # First, check bounds are respected
    valid = True
    for i in range(len(degenom)):
        if (type(degenom[i]) != bounds[i][2]) or (degenom[i] < bounds[i][0]) or (degenom[i] > bounds[i][1]):
            valid = False
    # Second, check constraints are respected
    if (degenom[1] - degenom[0] < 10):
        valid = False
    return valid



def gini_sharpe_fitness(degenom: list, bounds, fitness_settings: dict):
    """
    Measure how good a strategy is. The aim is to have the highest gini * sqrt(sharpe) possible.
    
    Arguments:
        degenom [list]: set of optimization parameters
        bounds [list of (lb, ub, type)]: lower bounds, upper bounds and type of optimization parameters
        fitness_settings [dict]: {
            "start_ts",
            "end_ts",
            "contract_type",
            "risk_factor",
            "timeframe",
            "universe",
            "initial_portfolio_value",
            "fee_rate"
        }
    """
    required_labels = [
        "start_ts", "end_ts", "contract_type",
        "risk_factor", "timeframe", "universe",
        "initial_portfolio_value", "fee_rate"
    ]
    if not all([required_labels[i] in fitness_settings.keys() for i in range(len(required_labels))]):
        raise Exception("[gini_sharpe_fitness()]: Wrong fitness settings passed.")
    start_ts = fitness_settings['start_ts']
    end_ts = fitness_settings['end_ts']
    contract_type = fitness_settings['contract_type']
    risk_factor = fitness_settings['risk_factor']
    timeframe = fitness_settings['timeframe']
    universe = fitness_settings['universe']
    initial_portfolio_value = fitness_settings['initial_portfolio_value']
    fee_rate = fitness_settings['fee_rate']
    # First, check bounds are respected
    if not is_valid(degenom):
        return 0.
    # Set the strategy
    strategy_settings = {
            'contract_type': contract_type, 
            'risk_factor': risk_factor,
            'fast_window': degenom[0],
            'slow_window': degenom[1],
            'std_window': degenom[2],
            'breakout_window': degenom[3],
            'exit_multiplier': degenom[4]
        }
    strategy = strategies.TrendFollowingStrategy(settings=strategy_settings)
    # Configure and run backtest
    backtest_settings = {
        'timeframe': timeframe,
        'universe': universe,
        'initial_portfolio_value': initial_portfolio_value,
        'fee_rate': fee_rate
    }
    backtest = backtester.Backtester(strategy, start_ts, end_ts, backtest_settings)
    opt_data = backtest.execute()
    roi = opt_data['annualized_roi']
    gini = opt_data['gini_coefficient']
    sharpe = opt_data['sharpe_ratio']
    fitness = np.max([0., (1-gini)*np.sqrt(sharpe)]) if (sharpe > 0 and roi > 0) else 0.
    return fitness



fitness_settings = {
    "start_ts": pd.Timestamp("2020-01-01 00:00:00"),
    "end_ts": pd.Timestamp("2021-01-01 00:00:00"),
    "contract_type": "long",
    "risk_factor": 0.001,
    "timeframe": '4h',
    "universe": ["BTCUSDT"],
    "initial_portfolio_value": 1000,
    "fee_rate": 0.0004
}

bounds = [
    (2, 200, int),
    (2, 200, int),
    (2, 200, int),
    (2, 200, int),
    (0.5, 10, float)
]

n_iter = 20
n_pop = 30
n_parents = 20
n_bests = 2
r_cross = 0.9
r_mut = 0.3

GO = GeneticOptimizer(
    objective=gini_sharpe_fitness,
    constraint=is_valid,
    bounds=bounds,
    fitness_settings=fitness_settings,
    n_iter=n_iter,
    n_pop=n_pop,
    n_parents=n_parents,
    n_bests=n_bests,
    r_cross=r_cross,
    r_mut=r_mut
)

initial_population = [
    [40, 80, 40, 50, 3],
    [10, 20, 30, 30, 2],
    [50, 100, 50, 50, 2],
    [58, 69, 2, 48, 0.596],
    [90, 113, 19, 2, 0.782],
    [58, 198, 19, 2, 0.782]
]

initial_population = [GO.normalize(degenom) for degenom in initial_population]
GO.search(initial_population)