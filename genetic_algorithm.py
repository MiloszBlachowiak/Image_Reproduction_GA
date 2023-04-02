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


class PixelsReproductionData:
    def __init__(self, image, iter_num, selection_methods, crossover_methods, mutation_methods, number_of_parents,
                 number_of_offsprings, mutation_percentage, epsilon, terminate_after):
        self.image = image
        self.iter_num = iter_num
        self.population_init_method = PopulationInitializationForPixels().random
        self.selection_methods = selection_methods
        self.crossover_methods = crossover_methods
        self.mutation_methods = mutation_methods
        self.number_of_parents = number_of_parents
        self.number_of_offsprings = number_of_offsprings
        self.mutation_percentage = mutation_percentage
        self.epsilon = epsilon
        self.terminate_after = terminate_after

class ImageReproductionForPixels:
    def __init__(self, pixelsReproductionData: PixelsReproductionData):
        self.data = pixelsReproductionData
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
        if fitness_value_new - fitness_value_best < self.data.epsilon:
            self.termination_cond_iter += 1
        else:
            self.termination_cond_iter = 0

        if self.termination_cond_iter == self.data.terminate_after:
            return True

        return False

    def perform_selection(self, population, qualities):
        selection_fun = np.random.choice(self.data.selection_methods)
        return selection_fun(population, qualities, self.data.number_of_parents)

    def perform_crossover(self, mating_pool):
        crossover_fun = np.random.choice(self.data.crossover_methods)
        return CrossoverForPixels.perform_crossover(mating_pool, crossover_fun, self.data.number_of_offsprings)

    def perform_mutation(self, population):
        mutation_fun = np.random.choice(self.data.mutation_methods)
        return mutation_fun(population, self.data.mutation_percentage)

    def reproduce_image(self):

        chromosome_base = imgRGB2chromosome(self.data.image)

        population = self.data.population_init_method(self.data.image.shape)

        best_chromosome = population[0]

        for iter in range(self.data.iter_num):

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

        best_reproduced_image = chromosome2imgRGB(best_chromosome, self.data.image.shape)

        return best_reproduced_image


class ImageReproductionForTriangles:
    def __init__(self):
        pass


def run_program():
    # read parameters
    reproduced_image_path = None
    image_path = "tangerines.jpg"
    img = cv2.imread(image_path)
    num_of_iterations = 15000
    selection_methods = [Selection.rank]  # list of lambdas
    crossover_methods = [CrossoverForPixels.single_point]
    mutation_methods = [MutationsForPixels.replacement]
    number_of_parents = 4
    number_of_offsprings = 8
    mutation_percentage = 0.01
    epsilon = 10 ** (-12)
    terminate_after = 500

    pixelsReproductionParams = PixelsReproductionData(img, num_of_iterations, selection_methods, crossover_methods,
                                                      mutation_methods, number_of_parents, number_of_offsprings,
                                                      mutation_percentage, epsilon, terminate_after)

    pixelsReproduction = ImageReproductionForPixels(pixelsReproductionParams)
    reproduced_image = pixelsReproduction.reproduce_image()

    cv2.imwrite('reproduced_image.jpg', reproduced_image)
