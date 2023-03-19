# file with genetic algorithm

import numpy as np
from typing import Tuple


def imgRGB2chromosome(img: np.array) -> np.array:
    return np.reshape(img, -1)


def chromosome2imgRGB(chromosome: np.array, img_shape: Tuple[int, ...]) -> np.array:
    return np.reshape(chromosome, img_shape)


def fitness_function(img_value: float, chromosome_value: float) -> float:
    return np.sum(img_value) - np.mean(np.abs(img_value - chromosome_value))


# jako parametry ogólniki: selection_method zamiast wypisania konkretnej metody
def reproduce_image(selection_method, mutation_method):
    raise NotImplementedError


# function which connects algorithm implementation with gui
def run_program():
    raise NotImplementedError