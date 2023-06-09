# source https://www.geeksforgeeks.org/crossover-in-genetic-algorithm/
import numpy as np
from itertools import combinations
from math import ceil


class Crossovers:
    def __init__(self):
        pass

    # single operation of single-point crossover
    @staticmethod
    def single_point(parent_one: np.ndarray, parent_two: np.ndarray):

        size = min(parent_one.size, parent_two.size)
        point = np.random.randint(1, size)
        
        offspring_one = np.append(parent_one[:point], parent_two[point:])
        offspring_two = np.append(parent_two[:point], parent_one[point:])

        return offspring_one, offspring_two

    # single operation of two-point crossover
    @staticmethod
    def two_point(parent_one: np.ndarray, parent_two: np.ndarray):
        size = min(parent_one.size, parent_two.size)

        # randomly picking two crossover points
        points = np.random.choice(range(1, size), 2, replace=False)
        points = sorted(points)

        offspring_one= np.append(np.append(parent_one[:points[0]], parent_two[points[0]:points[1]]), parent_one[points[1]:])
        offspring_two = np.append(np.append(parent_two[:points[0]], parent_one[points[0]:points[1]]), parent_two[points[1]:])

        return offspring_one, offspring_two

    # single operation of uniform crossover
    @staticmethod
    def uniform(parent_one: np.ndarray, parent_two: np.ndarray):
        size = min(parent_one.size, parent_two.size)

        # randomly picking half the genes' ids
        genes = np.random.choice(range(size), int(size/2), replace=False)
        genes = sorted(genes)

        offspring_one = np.copy(parent_one)
        offspring_two = np.copy(parent_two)

        crossover_mask = np.random.randint(0, 2, size)

        for i in range(size):
            if crossover_mask[i] == 1:
                offspring_one[i], offspring_two[i] = offspring_two[i], offspring_one[i]

        return offspring_one, offspring_two

    # function performing crossover operation over a given mating pool
    @staticmethod
    def perform_crossover(mating_pool: np.ndarray, crossover_method=single_point, number_of_offsprings: int=8) -> np.ndarray:
        offsprings = []

        # all permutations of parents' ids
        parents_pairs_permutations = list(combinations(range(len(mating_pool)), r=2))

        # randomly choosing uniqe pairs of parents' ids to cross over
        chosen_pairs_ids = np.random.choice(len(parents_pairs_permutations), ceil(number_of_offsprings / 2), replace=False)

        for pair_id in chosen_pairs_ids:
            parents_pair = parents_pairs_permutations[pair_id]

            offspring_one, offspring_two = crossover_method(mating_pool[parents_pair[0]], mating_pool[parents_pair[1]])

            offsprings.append(offspring_one)
            offsprings.append(offspring_two)

        # since every single crossover operation gives us 2 offsprings
        # if the given number of offsprings is odd, we have to remove random offspring from the list of offsprings
        if number_of_offsprings % 2 == 1:
            offsprings.pop(np.random.randint(len(offsprings)))

        return np.array(offsprings)
