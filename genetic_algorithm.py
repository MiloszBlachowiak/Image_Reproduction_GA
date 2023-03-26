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

def reproduce_image(image_path: str, iter_num: int, selection_method, crossover_method, mutation_method, mutation_percentage):
    img = cv2.imread(image_path)

    img_shape = img.shape

    chromosome_base = imgRGB2chromosome(img)

    population = population_initialization.random(img.shape)
   
    best_chromosome = population[0]

    for iter in range(iter_num):

        fitness_function_values = calculate_population_fitness(population, chromosome_base)
        
        best_chromosome_candidate = choose_best_chromosome(population, fitness_function_values)
        
        if fitness_function(chromosome_base, best_chromosome_candidate) > fitness_function(chromosome_base, best_chromosome):
            best_chromosome = best_chromosome_candidate

        mating_pool = selection_method(population, fitness_function_values, 4)

        population = crossover_method(mating_pool, number_of_offsprings=8)

        population = mutation_method(population, mutation_percentage)
        
        fitness_function_values = calculate_population_fitness(population, chromosome_base)
        
        best_chromosome_candidate = choose_best_chromosome(population, fitness_function_values)
        
        if fitness_function(chromosome_base, best_chromosome_candidate) > fitness_function(chromosome_base, best_chromosome):
            best_chromosome = best_chromosome_candidate
        
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
    num_of_iterations = 10000
    mutation_percentage = 0.01
    selection_method = selection.tournament
    crossover_method = crossover.perform_crossover
    mutation_method = mutation.random_change


    reproduced_image = reproduce_image(image_path, num_of_iterations, selection_method, crossover_method,
                                       mutation_method, mutation_percentage)

    cv2.imwrite('reproduced_image.jpg', reproduced_image)
