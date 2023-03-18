# file with genetic algorithm

def imgRGB2chromosome():
    raise NotImplementedError

def chromosome2imgRGB():
    raise NotImplementedError

def fitness_function():
    raise NotImplementedError

# jako parametry ogólniki: selection_method zamiast wypisania konkretnej metody
def reproduce_image(selection_method, mutation_method, ...):
    raise NotImplementedError


# function which connects algorithm implementation with gui
def run_program():
    raise NotImplementedError