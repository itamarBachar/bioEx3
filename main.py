import sys
import random
import numpy as np

from solution import Nn

MUTATION_RATE = 0.05
POPULATION_SIZE = 20
NUM_GENERATIONS = 350
REPLICATION = 0.1


def main(first_file, second_file):
    # Read the first file
    data = []
    label_data = []
    with open(first_file, 'r') as f:
        for line in f:
            bits, label = line.strip().split()  # Splitting by whitespace
            data.append(list(map(int, list(bits))))
            label_data.append(int(label))
    input_size = 16
    hidden_sizes = [128, 64, 32]
    # for i in range(POPULATION_SIZE):
    #     hidden_size_i = []
    #     # randaomly generate hidden layer sizes for random size of layers
    #     amount = random.randint(2, 5)
    #     size = random.randint(32, 64)
    #     hidden_size_i.append(size)
    #     for j in range(amount - 1):
    #         size = random.randint(int(size/2), size)
    #         hidden_size_i.append(size)
    #     hidden_sizes.append(hidden_size_i)
    output_size = 1
    train, test = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]
    train_label, test_label = label_data[:int(len(label_data) * 0.8)], label_data[int(len(label_data) * 0.8):]
    genetic_algorithm(train, train_label, input_size, hidden_sizes, output_size)


def crossover(winner, winner2):
    child_weights = []
    child_biases = []
    for i in range(len(winner.weights)):
        weight1 = winner.weights[i]
        weight2 = winner2.weights[i]
        child_weight = np.where(np.random.rand(*weight1.shape) < 0.5, weight1, weight2)
        child_weights.append(child_weight)
        bias1 = winner.biases[i]
        bias2 = winner2.biases[i]
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


def genetic_algorithm(train, train_label, input_size, hidden_sizes, output_size):
    population = []
    for i in range(POPULATION_SIZE):
        population.append(Nn(input_size, hidden_sizes, output_size, None, None))
        len(population[0].weights)
    offspring = []
    for generation in range(NUM_GENERATIONS):
        #  random select batch from training data
        batch_size = 100
        batch_index = random.sample(range(len(train)), batch_size)
        batch = [train[i] for i in batch_index]
        batch_label = [train_label[i] for i in batch_index]
        fitness_scores = [nn.evaluate_fitness(batch, batch_label) for nn in population]
        num_replication = int(REPLICATION * POPULATION_SIZE)
        fitness_scores_copy = fitness_scores.copy()
        for i in range(num_replication):
            index = fitness_scores.index(max(fitness_scores_copy))
            offspring.append(population[index])
            fitness_scores_copy[index] = -1
        for i in range(POPULATION_SIZE - num_replication):
            tournament_size = 5
            tournament = random.sample(population, tournament_size)
            # Choose the individual with the highest fitness score as the winner
            winner = max(tournament, key=lambda x: x.score)
            tournament = random.sample(population, tournament_size)
            # Choose the individual with the highest fitness score as the winner
            winner2 = max(tournament, key=lambda x: x.score)
            while winner == winner2:
                tournament = random.sample(population, tournament_size)
                winner2 = max(tournament, key=lambda x: x.score)
            child = crossover(winner, winner2)
            child = mutate(child, 0.05)
            offspring.append(child)
        population = offspring
        print("Generation: ", generation, "Best fitness: ", max(fitness_scores))


if __name__ == '__main__':
    # get the arguments passed to python main.py
    args = sys.argv
    main(args[1], args[2])
