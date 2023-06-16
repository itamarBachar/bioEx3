import sys
import numpy as np
from solution import Nn

output_size = 1
input_size = 16
MUTATION_RATE = 0.2
POPULATION_SIZE = 50
NUM_GENERATIONS = 150
REPLICATION = 0.1


def runnet0(weight_file, test_file):
    weight, bias = read_matrices_and_biases_from_file(weight_file)
    hidden_sizes = []
    for i in range(len(weight) - 1):
        hidden_sizes.append((weight[i].shape[1]))
    best_solution = Nn(input_size, hidden_sizes, output_size, weight, bias)
    with open(test_file, 'r') as file:
        input_data = file.read().splitlines()
    with open('output0.txt', 'w') as f:
        for x in input_data:
            # cast the str to list
            y = list(map(int, list(x)))
            output = best_solution.forward(y)
            if output > 0.5:
                f.write(x + "   1\n")
            else:
                f.write(x + "   0\n")


def read_matrices_and_biases_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matrices = {}
    biases = {}
    current_data = []
    current_name = None
    current_dictionary = None
    for line in lines:
        line = line.strip()
        if line.startswith("Matrix"):
            if current_name and current_dictionary is not None:
                current_dictionary[current_name] = current_data
            current_name = line
            current_data = []
            current_dictionary = matrices
        elif line.startswith("biases"):
            if current_name and current_dictionary is not None:
                current_dictionary[current_name] = current_data
            current_name = line
            current_data = []
            current_dictionary = biases
        elif line:
            current_data.append(list(map(float, line.split())))
    if current_name and current_dictionary is not None:
        current_dictionary[current_name] = current_data
    weights = []
    for key in matrices:
        matrices[key] = np.array(matrices[key])
        weights.append(matrices[key])
    ret_biases = []
    for key in biases:
        biases[key] = np.array(biases[key])
        ret_biases.append(biases[key])
    return weights, ret_biases


if __name__ == '__main__':
    runnet0(sys.argv[1], sys.argv[2])
