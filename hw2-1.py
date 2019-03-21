#!python3
"""Main program for homework 2-1."""
from argparse import ArgumentParser


def main():
    """Parse arguments."""
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
    print(args)


if __name__ == '__main__':
    main()
