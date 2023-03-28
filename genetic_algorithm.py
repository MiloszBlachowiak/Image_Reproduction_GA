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


def calculate_population_fitness(population, chromosome_base):
    fitness_function_values = np.zeros(population.shape[0])
    for individual_idx in range(population.shape[0]):
        chromosome_candidate = population[individual_idx]
        fitness_function_values[individual_idx] = fitness_function(chromosome_base, chromosome_candidate)
    return fitness_function_values


def choose_best_chromosome(population, qualities):
    best_idx = np.argmax(qualities)
    return population[best_idx]


def reproduce_image(image_path: str, iter_num: int, selection_function, crossover_function, crossover_method, mutation_function,
                    mutation_percentage, epsilon=10**(-12), terminate_after=500):
    img = cv2.imread(image_path)

    img_shape = img.shape

    chromosome_base = imgRGB2chromosome(img)

    population = population_initialization.random(img.shape)
   
    best_chromosome = population[0]

    termination_cond_iter = 0

    for iter in range(iter_num):

        fitness_function_values = calculate_population_fitness(population, chromosome_base)

        mating_pool = selection_function(population, fitness_function_values, 4)

        population = crossover_function(mating_pool, crossover_method, number_of_offsprings=8)

        population = mutation_function(population, mutation_percentage)
        
        fitness_function_values = calculate_population_fitness(population, chromosome_base)
        
        best_chromosome_candidate = choose_best_chromosome(population, fitness_function_values)
        
        fitness_value_best = fitness_function(chromosome_base, best_chromosome)
        fitness_value_new = fitness_function(chromosome_base, best_chromosome_candidate)

        if fitness_value_new > fitness_value_best:
            best_chromosome = best_chromosome_candidate

        # termination condition
        if fitness_value_new - fitness_value_best < epsilon:
            termination_cond_iter += 1
        else:
            termination_cond_iter = 0

        if termination_cond_iter == terminate_after:
            break
        
    # fitness_function_values = calculate_population_fitness(population, chromosome_base)
    # chromosome_candidate = choose_best_chromosome(population, fitness_function_values)
    # reproduced_image = chromosome2imgRGB(chromosome_candidate, img_shape)
    best_reproduced_image = chromosome2imgRGB(best_chromosome, img_shape)

    return best_reproduced_image


# function which connects algorithm implementation with gui
def run_program():
    # read parameters
    image_path = "tangerines.jpg"
    reproduced_image_path = None
    num_of_iterations = 15000
    mutation_percentage = 0.01
    selection_function = selection.tournament
    crossover_function = crossover.perform_crossover
    crossover_method = "single_point"
    mutation_function = mutation.random_swap


    reproduced_image = reproduce_image(image_path, num_of_iterations, selection_function, crossover_function,
                                       crossover_method, mutation_function, mutation_percentage)

    cv2.imwrite('reproduced_image.jpg', reproduced_image)
