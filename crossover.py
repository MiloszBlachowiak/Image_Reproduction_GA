# source https://www.geeksforgeeks.org/crossover-in-genetic-algorithm/
import numpy as np
from itertools import combinations
from math import ceil
   

# single operation of single-point crossover    
def single_point(parent_one: np.ndarray, parent_two: np.ndarray):

    size = parent_one.size
    point = np.random.randint(1, size)

    offspring_one = np.append(parent_one[:point], parent_two[point:])
    offspring_two = np.append(parent_two[:point], parent_one[point:])

    return offspring_one, offspring_two


# single operation of two-point crossover
def two_point(parent_one: np.ndarray, parent_two: np.ndarray):
    size = parent_one.size

    # randomly picking two crossover points
    points = np.random.choice(range(1, size), 2, replace=False)
    points = sorted(points)

    offspring_one= np.append(np.append(parent_one[:points[0]], parent_two[points[0]:points[1]]), parent_one[points[1]:])
    offspring_two = np.append(np.append(parent_two[:points[0]], parent_one[points[0]:points[1]]), parent_two[points[1]:])

    return offspring_one, offspring_two


def uniform():
    raise NotImplementedError


# function performing crossover operation over a given mating pool
def perform_crossover(mating_pool: np.ndarray, method: str="single_point", number_of_offsprings: int=8) -> np.ndarray:
    offsprings = []

    # all permutations of parents' ids
    parents_pairs_permutations = list(combinations(range(len(mating_pool)), r=2))

    # randomly choosing uniqe pairs of parents' ids to cross over
    chosen_pairs_ids = np.random.choice(len(parents_pairs_permutations), ceil(number_of_offsprings / 2), replace=False)
    
    for pair_id in chosen_pairs_ids:
        parents_pair = parents_pairs_permutations[pair_id]

        if method == "single_point":
            offspring_one, offspring_two = single_point(mating_pool[parents_pair[0]], mating_pool[parents_pair[1]])
        elif method == "two_point":
            offspring_one, offspring_two = two_point(mating_pool[parents_pair[0]], mating_pool[parents_pair[1]])
    
        offsprings.append(offspring_one)
        offsprings.append(offspring_two)

    # since every single crossover operations gives us 2 offsprings
    # if the given number of offsprings is odd, we have to remove random offspring from the list of offsprings
    if number_of_offsprings % 2 == 1:
        offsprings.pop(np.random.randint(len(offsprings)))
    
    return np.array(offsprings)
