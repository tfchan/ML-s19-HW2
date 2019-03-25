#!python3
"""Main program for homework 2-1."""
from argparse import ArgumentParser
import struct
import numpy as np
import naive_bayes as nb


def read_mnist(file):
    """Read mnist file and return data with appropriate shape."""
    with open(file, 'rb') as f:
        dimension = struct.unpack_from('>B', f.read(4), 3)[0]
        shape = []
        for _ in range(dimension):
            shape += [struct.unpack('>I', f.read(4))[0]]
        data = np.fromfile(f, dtype='uint8').reshape(tuple(shape))
    return data


def imgs2features(imgs):
    """Convert array of images to array of features."""
    return imgs.reshape((imgs.shape[0], imgs.shape[1] * imgs.shape[2]))


def print_predictions(pred_results, label):
    """Print prediction result in a readable way."""
    error_count = 0
    for i, pred_result in enumerate(pred_results):
        print('Posterior (in log scale):')
        for class_, proba in pred_result.items():
            print(f'{class_}: {proba}')
        pred_num = min(pred_result, key=lambda x: pred_result[x])
        print(f'Prediction: {pred_num}, Ans:{label[i]}\n')
        if pred_num != label[i]:
            error_count += 1
    return error_count / len(pred_results)


def print_imaginations(imaginations):
    """Print imaginations for each class."""
    img_size = (28, 28)
    for class_, feature in imaginations.items():
        img = np.array(feature).reshape(img_size)
        print(f'{class_}:')
        for row in range(img.shape[0]):
            print(*img[row], sep=' ')
        print()


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
    parser.add_argument('mode', type=int, choices=[0, 1],
                        help='0 or 1 (Discrete or continuous mode)')
    args = parser.parse_args()

    # Read mnist data
    train_image = read_mnist(args.train_image_path)
    train_label = read_mnist(args.train_label_path)
    test_image = read_mnist(args.test_image_path)
    test_label = read_mnist(args.test_label_path)

    # Train the model
    nbc = nb.DiscreteNB()
    nbc.fit(imgs2features(train_image), train_label)

    # Predict testing data
    prediction = nbc.predict_log_proba(imgs2features(test_image))
    error_rate = print_predictions(prediction, test_label)

    # Print imaginary
    imaginations = nbc.get_imagination()
    print_imaginations(imaginations)
    print(f'Error rate:{error_rate}')


if __name__ == '__main__':
    main()
