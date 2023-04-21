# file with genetic algorithm

import cv2
import numpy as np
from typing import Tuple

from selections import Selections
from crossover import Crossovers
from mutation import Mutations
from population_initialization import PopulationInitializations
from triangulation import Triangulation


class ReproductionData:
    def __init__(self, image, iter_num, selection_methods, crossover_methods, mutation_methods, number_of_parents,
                 number_of_offsprings, mutation_percentage, epsilon, terminate_after, enable_triangulation, n_points=None):
        self.image = image
        self.iter_num = iter_num
        self.population_init_method = PopulationInitializations().random
        self.selection_methods = selection_methods
        self.crossover_methods = crossover_methods
        self.mutation_methods = mutation_methods
        self.number_of_parents = number_of_parents
        self.number_of_offsprings = number_of_offsprings
        self.mutation_percentage = mutation_percentage
        self.epsilon = epsilon
        self.terminate_after = terminate_after
        self.enable_triangulation = enable_triangulation

        self.n_points = n_points
        self.image_tri = None
        self.triangle_weights = None

    def log_data(self):
        print("iter_num: ", self.iter_num)
        print("population_init_method: ", self.population_init_method)
        print("selection_methods: ", self.selection_methods)
        print("crossover_methods: ", self.crossover_methods)
        print("mutation_methods: ", self.mutation_methods)
        print("number_of_parents: ", self.number_of_parents)
        print("number_of_offsprings: ", self.number_of_offsprings)
        print("mutation_percentage: ", self.mutation_percentage)
        print("epsilon: ", self.epsilon)
        print("terminate_after: ", self.terminate_after)
        print("enable_triangulation: ", self.enable_triangulation)
        print("n_points: ", self.n_points)
        print("image_tri: ", self.image_tri)
        print("triangle_weights: ", self.triangle_weights)


class ImageReproduction:
    def __init__(self, pixelsReproductionData: ReproductionData):
        self.data = pixelsReproductionData
        self.termination_cond_iter = 0

    def triangulate_image(self):
        self.data.image_tri = Triangulation()
        self.data.image_tri.triangulate(self.data.image, self.data.n_points)
        weights = self.data.image_tri.get_triangle_weights(self.data.image)
        weights = weights / np.max(weights) #normalization
        self.data.triangle_weights = np.concatenate([[weights[i]] * 3 for i in range(len(weights))]) # 3x because of R, G and B components

    def imgRGB2chromosome(self) -> np.array:
        if self.data.enable_triangulation:
            self.triangulate_image()
            colours = self.data.image_tri.get_triangle_colour(self.data.image)
            return np.reshape(colours, -1)
        else:
            return np.reshape(self.data.image, -1)

    def chromosome2imgRGB(self, chromosome) -> np.array:
        if self.data.enable_triangulation:
            colours = np.reshape(chromosome, (-1, 3))

            final_image = np.zeros(self.data.image.shape)
            simplex = self.data.image_tri.get_simplex(self.data.image)
            pixel_coords = self.data.image_tri.get_pixel_coords(self.data.image.shape[:2])
            for idx, coord in enumerate(pixel_coords):
                triangle_id = simplex[idx]
                colour = colours[triangle_id, :]
                final_image[coord[1], coord[0], :] = colour
            return final_image
        else:
            return np.reshape(chromosome, self.data.image.shape)

    def choose_best_chromosome(self, population, qualities):
        best_idx = np.argmax(qualities)
        return population[best_idx]

    def fitness_function_for_shapes(self, chromosome_base, chromosome_candidate):
        if self.data.enable_triangulation:
            weights = self.data.triangle_weights
            return np.sum(chromosome_base * weights) - np.mean(np.abs(chromosome_base - chromosome_candidate) * weights)
        else:
            return np.sum(chromosome_base) - np.mean(np.abs(chromosome_base - chromosome_candidate))

    def calculate_population_fitness(self, population, chromosome_base):
        fitness_function_values = np.zeros(population.shape[0])
        for individual_idx in range(population.shape[0]):
            chromosome_candidate = population[individual_idx]
            fitness_function_values[individual_idx] = self.fitness_function_for_shapes(chromosome_base, chromosome_candidate)
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
        return Crossovers.perform_crossover(mating_pool, crossover_fun, self.data.number_of_offsprings)

    def perform_mutation(self, population):
        mutation_fun = np.random.choice(self.data.mutation_methods)
        return mutation_fun(population, self.data.mutation_percentage)

    def reproduce_image(self):

        print("Started the algorithm...")

        chromosome_base = self.imgRGB2chromosome()    

        population = self.data.population_init_method(len(chromosome_base), self.data.number_of_offsprings)
        best_chromosome = population[0]

        for iter in range(self.data.iter_num):

            fitness_function_values = self.calculate_population_fitness(population, chromosome_base)

            mating_pool = self.perform_selection(population, fitness_function_values)
            population = self.perform_crossover(mating_pool)
            population = self.perform_mutation(population)

            fitness_function_values = self.calculate_population_fitness(population, chromosome_base)

            best_chromosome_candidate = self.choose_best_chromosome(population, fitness_function_values)

            fitness_value_best = self.fitness_function_for_shapes(chromosome_base, best_chromosome)
            fitness_value_new = self.fitness_function_for_shapes(chromosome_base, best_chromosome_candidate)


            if fitness_value_new > fitness_value_best:
                best_chromosome = best_chromosome_candidate

            # termination condition
            if (self.is_termination_condition_fulfilled(fitness_value_new, fitness_value_best)):
                break

        # save original triangulated image (for comparison)
        if self.data.enable_triangulation:
            original_triangulated = self.chromosome2imgRGB(chromosome_base)
            cv2.imwrite('original_triangulated.jpg', original_triangulated)

        best_reproduced_image = self.chromosome2imgRGB(best_chromosome)

        print("Finished!")

        return best_reproduced_image


def run_program():
    # read parameters
    reproduced_image_path = None
    image_path = "tangerines.jpg"
    img = cv2.imread(image_path)
    num_of_iterations = 10000
    selection_methods = [Selections.tournament]  # list of lambdas
    crossover_methods = [Crossovers.single_point]
    mutation_methods = [Mutations.random_swap]

    number_of_parents = 4
    number_of_offsprings = 8

    mutation_percentage = 0.01
    epsilon = 10 ** (-12)
    terminate_after = 500

    # Triangularization parameters
    enable_triangulation = True
    n_points = 1800

    pixelsReproductionParams = ReproductionData(img, num_of_iterations, selection_methods, crossover_methods,
                                                mutation_methods, number_of_parents, number_of_offsprings,
                                                mutation_percentage, epsilon, terminate_after, enable_triangulation, n_points)

    pixelsReproduction = ImageReproduction(pixelsReproductionParams)
    reproduced_image = pixelsReproduction.reproduce_image()

    cv2.imwrite('reproduced_image.jpg', reproduced_image)
