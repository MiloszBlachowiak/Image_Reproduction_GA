# https://sound.eti.pg.gda.pl/student/isd/isd03-algorytmy_genetyczne.pdf
import numpy as np
    

class MutationsForPixels:
    def __init__(self):
        pass

    def replacement(self, population: np.ndarray, mutation_percent: float) -> np.ndarray:
        for idx in range(population.shape[0]):
            rand_idx = np.uint32(np.random.random(size=np.uint32(mutation_percent/100*population.shape[1]))
                                                        *population.shape[1])
            new_values = np.uint8(np.random.random(size=rand_idx.shape[0])*256)
            population[idx, rand_idx] = new_values
        return population

    def random_swap(self, population: np.ndarray, mutation_percent: float) -> np.ndarray:
        for idx in range(population.shape[0]):
            rand_idx_1 = np.uint32(np.random.random(size=np.uint32(mutation_percent/100*population.shape[1]))
                                                        *population.shape[1])
            rand_idx_2 = np.uint32(np.random.random(size=np.uint32(mutation_percent/100*population.shape[1]))
                                                        *population.shape[1])
            population[idx, rand_idx_1], population[idx, rand_idx_2] = population[idx, rand_idx_2], population[idx, rand_idx_1]
        return population

    def adjacent_swap(self, population: np.ndarray, mutation_percent: float) -> np.ndarray:
        for idx in range(population.shape[0]):
            rand_idx_1 = np.uint32(np.random.random(size=np.uint32(mutation_percent/100*population.shape[1]))
                                                        *population.shape[1])
            if np.any(rand_idx_1 == population.shape[1]-1):
                ind = np.where(rand_idx_1 == population.shape[1]-1)
                rand_idx_1 = np.delete(rand_idx_1, ind)
            idx_2 = rand_idx_1 + 1
            population[idx, rand_idx_1], population[idx, idx_2] = population[idx, idx_2], population[idx, rand_idx_1]
        return population

    def end_for_end_swap(self, population: np.ndarray, mutation_percent: float) -> np.ndarray:
        for idx in range(population.shape[0]):
            if np.random.random(size=1)[0] <= mutation_percent:
                rand_idx = np.uint32(np.random.random(size=1) * population.shape[1])[0]
                population[idx] = np.array(list(population[idx, rand_idx:]) + list(population[idx, :rand_idx]))
        return population

    def inversion(self, population: np.ndarray, mutation_percent: float) -> np.ndarray:
        for idx in range(population.shape[0]):
            if np.random.random(size=1)[0] <= mutation_percent:
                rand_idx = np.uint32(np.random.random(size=2) * population.shape[1])
                first, second = min(rand_idx), max(rand_idx)
                population[idx, first:second] = np.array(list(population[idx, first:second])[::-1])
        return population


class MutationsForTriangles:
    def __init__(self):
        pass
