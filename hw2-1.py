#!python3
"""Main program for homework 2-1."""
from argparse import ArgumentParser
import struct
import numpy as np


def read_mnist(file):
    """Read mnist file and return data with appropriate shape."""
    with open(file, 'rb') as f:
        dimension = struct.unpack_from('>B', f.read(4), 3)[0]
        shape = []
        for _ in range(dimension):
            shape += [struct.unpack('>I', f.read(4))[0]]
        data = np.fromfile(f, dtype='uint8').reshape(tuple(shape))
    return data


def main():
    """Perform main task of the program."""
    # Parse arguments
    parser = ArgumentParser(description='Naive Bayes classifier for MNIST')
    parser.add_argument('train_image_path', type=str,
                        help='Path to training image data')
    parser.add_argument('train_label_path', type=str,
                        help='Path to training label data')
    parser.add_argument('test_image_path', type=str,
                        help='Path to testing image data')
    parser.add_argument('test_label_path', type=str,
                        help='Path to testing label data')
    parser.add_argument('mode', type=int, help='Discrete or continuous mode')
    args = parser.parse_args()

    # Read mnist data
    train_image = read_mnist(args.train_image_path)
    train_label = read_mnist(args.train_label_path)
    test_image = read_mnist(args.test_image_path)
    test_label = read_mnist(args.test_label_path)


if __name__ == '__main__':
    main()
