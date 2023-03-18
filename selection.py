import numpy as np

# source: https://rocreguant.com/roulette-wheel-selection-python/2019/
def roulette_wheel(population, qualities, number_of_parents):
    population_fitness = np.sum(qualities)
    individual_probabilities = qualities / population_fitness
    return np.random.choice(population, size=number_of_parents, replace=False, p=individual_probabilities)


def rank(population, qualities, number_of_parents):
    ranking_id = np.argsort(qualities)
    return population[ranking_id][:number_of_parents]


def tournament(population, qualities, number_of_parents):
    parents = np.zeros(number_of_parents)
    population_groups = np.array_split(population, number_of_parents)
    qualities_groups = np.array_split(qualities, number_of_parents)
    for group_id in range(number_of_parents):
        population_group = population_groups[group_id]
        qualities_group = qualities_groups[group_id]
        max_quality_idx = np.argmax(qualities_group)
        parents[group_id] = population_group[max_quality_idx]
    return parents


def elitist(population, qualities, number_of_parents):
    raise NotImplementedError
