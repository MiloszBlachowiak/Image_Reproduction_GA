# https://sound.eti.pg.gda.pl/student/isd/isd03-algorytmy_genetyczne.pdf
import numpy as np

# short description
def mutationName():
    raise NotImplementedError
    
def replacement():
    raise NotImplementedError


def random_change(population, mutation_percent):
    for idx in range(population.shape[0]):
        
        rand_idx = np.uint32(np.random.random(size=np.uint32(mutation_percent/100*population.shape[1]))
                                                    *population.shape[1])
        new_values = np.uint8(np.random.random(size=rand_idx.shape[0])*256)
        population[idx, rand_idx] = new_values
    return population


def adjacent_swap():
    raise NotImplementedError


def end_for_end_swap():
    raise NotImplementedError


def inversion():
    raise NotImplementedError

