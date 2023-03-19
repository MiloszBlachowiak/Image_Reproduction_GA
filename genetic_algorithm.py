# file with genetic algorithm

import cv2
import numpy as np
from typing import Tuple

import crossover
import mutation
import population_initialization


def imgRGB2chromosome(img: np.array) -> np.array:
    return np.reshape(img, -1)


def chromosome2imgRGB(chromosome: np.array, img_shape: Tuple[int, ...]) -> np.array:
    return np.reshape(chromosome, img_shape)


def fitness_function(img_array: np.array, chromosome_array: np.array) -> float:
    return np.sum(img_array) - np.mean(np.abs(img_array - chromosome_array))


# jako parametry ogólniki: selection_method zamiast wypisania konkretnej metody
def reproduce_image(image_path: str, reproduced_image_path: str,  iter_num: int,
                    selection_method: str, mutation_method: str) -> None:
    img = cv2.imread(image_path)

    img_shape = img.shape

    chromosome = imgRGB2chromosome(img)

    population = population_initialization.random(chromosome)

    for iter in range(iter_num):

        fitness_function_values = []
        for individual in range(population.shape[0]):
            fitness_function_values[individual] = (individual, fitness_function(img, population[individual, :]))

        fitness_function_highest_values = [i for i in sorted(fitness_function_values, key=lambda x: x[1])[0]]

        population = crossover.perform_crossover(population[fitness_function_highest_values[5], :])

        population = mutation.replacement(population)

    reproduced_image = chromosome2imgRGB(population, img_shape)

    cv2.imwrite('reproduced_image', reproduced_image)


# function which connects algorithm implementation with gui
def run_program():
    raise NotImplementedError