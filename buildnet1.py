import sys
import random
import numpy as np

from solution import Nn

output_size = 1
input_size = 16
MUTATION_RATE = 0.2
POPULATION_SIZE = 100
NUM_GENERATIONS = 200
REPLICATION = 0.1


# split the nn0.txt to 2 files one is 80 percent and the other is 20 percent
def split_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    train = lines[:int(len(lines) * 1)]
    # test = lines[int(len(lines) * 0.8):]
    return train


def buildnet1(train_data, test_data):
    train = []
    train_label = []
    for line in train_data:
        bits, label = line.strip().split()
        train.append(list(map(int, list(bits))))
        train_label.append(int(label))
    test = []
    test_label = []
    for line in test_data:
        bits, label = line.strip().split()
        test.append(list(map(int, list(bits))))
        test_label.append(int(label))
    hidden_sizes = []
    for i in range(POPULATION_SIZE):
        hidden_size_i = []
        # randaomly generate hidden layer sizes for random size of layers
        amount = random.randint(1, 5)
        size = random.randint(25, 64)
        hidden_size_i.append(size)
        for j in range(amount - 1):
            size = random.randint(int(size / 2), size)
            hidden_size_i.append(size)
        hidden_sizes.append(hidden_size_i)
    solution = genetic_algorithm(train, train_label, test, test_label, input_size, hidden_sizes, output_size)
    # open file to write the solution
    with open("wnet1.txt", 'w') as file:
        for i, matrix in enumerate(solution.weights):
            file.write(f"Matrix {i + 1}:\n")
            for row in matrix:
                file.write(" ".join(str(element) for element in row) + "\n")
            file.write("\n")  # Add a blank line between matrices
        for i, matrix in enumerate(solution.biases):
            file.write(f"biases {i + 1}:\n")
            file.write(" ".join(str(element) for element in matrix) + "\n")
            file.write("\n")


def crossover(winner, winner2):
    child_weights = []
    child_biases = []
    # take the min len of the two parents
    min_len = min(len(winner.weights), len(winner2.weights))
    for i in range(min_len):
        weight1 = winner.weights[i]
        weight2 = winner2.weights[i]
        if weight1.shape != weight2.shape:
            # Handle variable-sized weights
            min_shape = np.minimum(weight1.shape, weight2.shape)
            min_shape = tuple(map(int, min_shape))  # Convert to integers
            child_weight = np.empty(min_shape)
            for index in np.ndindex(min_shape):
                child_weight[index] = np.random.choice([weight1[index], weight2[index]])
            child_weights.append(child_weight)
        else:
            child_weight = np.where(np.random.rand(*weight1.shape) < 0.5, weight1, weight2)
            child_weights.append(child_weight)

        bias1 = winner.biases[i]
        bias2 = winner2.biases[i]
        if bias1.shape != bias2.shape:
            # Handle variable-sized biases
            min_shape = np.minimum(bias1.shape, bias2.shape)
            min_shape = tuple(map(int, min_shape))  # Convert to integers
            child_bias = np.empty(min_shape)
            for index in np.ndindex(min_shape):
                child_bias[index] = np.random.choice([bias1[index], bias2[index]])
            child_biases.append(child_bias)
        else:
            child_bias = np.where(np.random.rand(*bias1.shape) < 0.5, bias1, bias2)
            child_biases.append(child_bias)

    return Nn(winner.input_size, winner.hidden_sizes, winner.output_size, child_weights, child_biases)


def mutate(child, mutation_range):
    for i in range(len(child.weights)):
        # Mutate weights
        mask = np.random.rand(*child.weights[i].shape) < MUTATION_RATE
        child.weights[i][mask] += np.random.uniform(-mutation_range, mutation_range, size=child.weights[i].shape)[mask]
        # Mutate biases
        mask = np.random.rand(*child.biases[i].shape) < MUTATION_RATE
        child.biases[i][mask] += np.random.uniform(-mutation_range, mutation_range, size=child.biases[i].shape)[mask]
        return child


def genetic_algorithm(train, train_label, test, test_label, input_size, hidden_sizes, output_size):
    population = []
    for i in range(POPULATION_SIZE):
        population.append(Nn(input_size, hidden_sizes[i], output_size, None, None))
        len(population[0].weights)
    best_solution = None
    for generation in range(NUM_GENERATIONS):
        offspring = []
        fitness_scores = [nn.evaluate_fitness(train, train_label) for nn in population]
        num_replication = int(REPLICATION * POPULATION_SIZE)
        fitness_scores_copy = fitness_scores.copy()
        for i in range(num_replication):
            index = fitness_scores.index(max(fitness_scores_copy))
            offspring.append(population[index])
            fitness_scores_copy[index] = -1
            if i == 0:
                best_solution = offspring[0].copy()
        for i in range(POPULATION_SIZE - num_replication):
            tournament_size = 3
            tournament = random.sample(population, tournament_size)
            # Choose the individual with the highest fitness score as the winner
            winner = max(tournament, key=lambda x: x.score)
            tournament = random.sample(population, tournament_size)
            # Choose the individual with the highest fitness score as the winner
            winner2 = max(tournament, key=lambda x: x.score)
            child = crossover(winner, winner2)
            child = mutate(child, 0.1)
            offspring.append(child)
        population = offspring
        print("Generation: ", generation, "Best fitness: ", max(fitness_scores))
    print("Test Accuracy" , best_solution.test_accuracy(test, test_label))
    return best_solution


if __name__ == '__main__':
    # get the arguments passed to python buildnet0.py
    args = sys.argv
    train = split_file(args[1])
    test = split_file(args[2])
    buildnet1(train, test)
