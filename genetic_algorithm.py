# file with genetic algorithm

import cv2
import numpy as np
from typing import Tuple

import crossover
import mutation
import population_initialization
import selection


def imgRGB2chromosome(img: np.array) -> np.array:
    return np.reshape(img, -1)


def chromosome2imgRGB(chromosome: np.array, img_shape: Tuple[int, ...]) -> np.array:
    return np.reshape(chromosome, img_shape)


def fitness_function(img_array: np.array, chromosome_array: np.array) -> float:
    return np.sum(img_array) - np.mean(np.abs(img_array - chromosome_array))


# jako parametry ogólniki: selection_method zamiast wypisania konkretnej metody
def reproduce_image(image_path: str, iter_num: int, reproduced_image_path: str=None,  
                    selection_method: str=None, mutation_method: str=None) -> None:
    img = cv2.imread(image_path)

    img_shape = img.shape

    chromosome = imgRGB2chromosome(img)

    population = population_initialization.random(img.shape)


    for iter in range(iter_num):

        fitness_function_values = np.zeros(population.shape[0])
        for individual_idx in range(population.shape[0]):
            fitness_function_values[individual_idx] = fitness_function(chromosome, population[individual_idx])

        mating_pool = selection.tournament(population, fitness_function_values, 4)

        population = crossover.perform_crossover(mating_pool, number_of_offsprings=8)

        population = mutation.random_change(population, 0.01)

    fitness_function_values = np.zeros(population.shape[0])
    for individual_idx in range(population.shape[0]):
        fitness_function_values[individual_idx] = fitness_function(chromosome, population[individual_idx])

    chromosome = selection.rank(population, fitness_function_values, 1)
    reproduced_image = chromosome2imgRGB(chromosome, img_shape)

    cv2.imwrite('reproduced_image.jpg', reproduced_image)


# function which connects algorithm implementation with gui
def run_program():
    raise NotImplementedError