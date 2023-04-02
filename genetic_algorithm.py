# file with genetic algorithm

import cv2
import numpy as np
from typing import Tuple

from selection import Selection
from crossover import CrossoverForPixels, CrossoverForTriangles
from mutation import MutationsForPixels, MutationsForTriangles
from population_initialization import PopulationInitializationForPixels, PopulationInitializationForTriangles


def imgRGB2chromosome(img: np.array) -> np.array:
    return np.reshape(img, -1)


def chromosome2imgRGB(chromosome: np.array, img_shape: Tuple[int, ...]) -> np.array:
    return np.reshape(chromosome, img_shape)


def choose_best_chromosome(population, qualities):
    best_idx = np.argmax(qualities)
    return population[best_idx]


class ImageReproductionForPixels:
    def __init__(self, iter_num, mutation_percentage, epsilon=10**(-12), terminate_after=500):
        self.population_init_functions = PopulationInitializationForPixels()
        self.selection_functions = Selection()
        self.crossover_methods = CrossoverForPixels()
        self.mutation_functions = MutationsForPixels()

        self.number_of_offsprings = 8
        self.number_of_parents = 4
        self.iter_num = iter_num
        self.mutation_percentage = mutation_percentage
        self.epsilon = epsilon
        self.terminate_after = terminate_after

        self.termination_cond_iter = 0

    def fitness_function(self, img_array: np.array, chromosome_array: np.array) -> float:
        return np.sum(img_array) - np.mean(np.abs(img_array - chromosome_array))

    def calculate_population_fitness(self, population, chromosome_base):
        fitness_function_values = np.zeros(population.shape[0])
        for individual_idx in range(population.shape[0]):
            chromosome_candidate = population[individual_idx]
            fitness_function_values[individual_idx] = self.fitness_function(chromosome_base, chromosome_candidate)
        return fitness_function_values

    def is_termination_condition_fulfilled(self, fitness_value_new, fitness_value_best):
        if fitness_value_new - fitness_value_best < self.epsilon:
            self.termination_cond_iter += 1
        else:
            self.termination_cond_iter = 0

        if self.termination_cond_iter == self.terminate_after:
            return True

        return False

    def perform_selection(self, population, qualities):
        return self.selection_functions.rank(population, qualities, self.number_of_parents)

    def perform_crossover(self, mating_pool: np.ndarray):
        method = "single_point"
        return self.crossover_methods.perform_crossover(mating_pool, method, self.number_of_offsprings)

    def perform_mutation(self, population):
        return self.mutation_functions.replacement(population, self.mutation_percentage)

    def reproduce_image(self, image_path: str):
        img = cv2.imread(image_path)

        img_shape = img.shape

        chromosome_base = imgRGB2chromosome(img)

        population = self.population_init_functions.random(img.shape)

        best_chromosome = population[0]

        for iter in range(self.iter_num):

            fitness_function_values = self.calculate_population_fitness(population, chromosome_base)

            mating_pool = self.perform_selection(population, fitness_function_values)

            population = self.perform_crossover(mating_pool)

            population = self.perform_mutation(population)

            fitness_function_values = self.calculate_population_fitness(population, chromosome_base)

            best_chromosome_candidate = choose_best_chromosome(population, fitness_function_values)

            fitness_value_best = self.fitness_function(chromosome_base, best_chromosome)
            fitness_value_new = self.fitness_function(chromosome_base, best_chromosome_candidate)

            if fitness_value_new > fitness_value_best:
                best_chromosome = best_chromosome_candidate

            # termination condition
            if (self.is_termination_condition_fulfilled(fitness_value_new, fitness_value_best)):
                break

        best_reproduced_image = chromosome2imgRGB(best_chromosome, img_shape)

        return best_reproduced_image


class ImageReproductionForTriangles:
    def __init__(self):
        pass


def run_program():
    # read parameters
    image_path = "tangerines.jpg"
    reproduced_image_path = None
    num_of_iterations = 15000
    mutation_percentage = 0.01

    pixelsReproduction = ImageReproductionForPixels(num_of_iterations, mutation_percentage)
    reproduced_image = pixelsReproduction.reproduce_image(image_path)

    cv2.imwrite('reproduced_image.jpg', reproduced_image)
