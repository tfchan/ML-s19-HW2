#!python3
"""Main program for homework 2-2."""
from argparse import ArgumentParser
from math import factorial


def read_file(file_path):
    """Read a given file and all lines."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def perform_online_learning(data, beta_params):
    """Do online learning on data with initial beta distribution params."""
    beta_prior = beta_params
    for case, outcomes in enumerate(data):
        # Calculte likelihood and posterior parameters
        n_toss = len(outcomes)
        n_head = outcomes.count('1')
        n_tail = n_toss - n_head
        p = n_head / n_toss
        n_combination = (factorial(n_toss)
                         / (factorial(n_head) * factorial(n_tail)))
        likelihood = n_combination * p ** n_head * (1 - p) ** n_tail
        beta_poste = (beta_prior[0] + n_head, beta_prior[1] + n_tail)

        # Print result
        print(f'Case {case}: {outcomes}')
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior: a = {beta_prior[0]} b = {beta_prior[1]}')
        print(f'Beta posterior: a = {beta_poste[0]} b = {beta_poste[1]}\n')

        # Update parameters
        beta_prior = beta_poste


def main():
    """Perform main task of the program."""
    # Parse arguments
    parser = ArgumentParser(description='Online learning of beta distribution')
    parser.add_argument('file_path', type=str,
                        help='Path to input file')
    parser.add_argument('beta_a', type=int,
                        help='Parameter a for initial beta prior')
    parser.add_argument('beta_b', type=int,
                        help='Parameter b for initial beta prior')
    args = parser.parse_args()

    # Read data from file
    data = read_file(args.file_path)

    # Perform online learning on the coin tossing data
    initial_beta_params = (args.beta_a, args.beta_b)
    perform_online_learning(data, initial_beta_params)


if __name__ == '__main__':
    main()
