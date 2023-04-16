# https://medium.datadriveninvestor.com/population-initialization-in-genetic-algorithms-ddb037da6773

import numpy as np


class PopulationInitializations:
    def __init__(self):
        pass

    def random(self, size_chromosom, number_individuals=10):

        # Creating an empty list to hold individuals
        population = np.empty(shape=(number_individuals, size_chromosom), dtype=np.uint8)

        # Filling individuals with random data
        for indv_num in range(number_individuals):
             population[indv_num] = np.random.random(size_chromosom)*256

        return population
