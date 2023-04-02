# https://medium.datadriveninvestor.com/population-initialization-in-genetic-algorithms-ddb037da6773

import numpy as np


class PopulationInitializationForPixels:
    def __init__(self):
        pass

    def random(self, img_size, number_individuals=10):
        if len(img_size) == 2:
          # For grayscale images
          size_chromosom = img_size[0]*img_size[1]
        else:
          # For color images
          size_chromosom = img_size[0]*img_size[1]*3

        # Creating an empty list to hold individuals
        population = np.empty(shape=(number_individuals, size_chromosom), dtype=np.uint8)

        # Filling individuals with random data
        for indv_num in range(number_individuals):
             population[indv_num] = np.random.random(size_chromosom)*256

        return population


class PopulationInitializationForTriangles:
    def __init__(self):
        pass
