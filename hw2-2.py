#!python3
"""Main program for homework 2-2."""
from argparse import ArgumentParser


def read_file(file_path):
    """Read a given file and return list containing list of char."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data = [list(line.strip()) for line in lines]
    return data


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


if __name__ == '__main__':
    main()
