import logging

import numpy as np

from data.perceptron import train, test

logging.basicConfig(
    format='[%(asctime)s](%(levelname)s) %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S'
)

w = np.random.rand(4)
n = .01
epochs = np.inf

logging.info('perceptron inputs configured')


def perceptron():
    logging.debug(f'w={w}')

    epoch = 0

    while True:
        error = False

        for sample in train:
            u = sum(sample[:-1] * w)
            y = 1 if u >= 0 else -1
            d = sample[-1]
            error = y != d

            if error:
                for j in range(len(w)):
                    w[j] += n * (d - y) * sample[j]
        epoch += 1

        if epoch > epochs or not error:
            logging.info(f'total epochs: {epoch}')
            logging.debug(f'w={w}')
            break


def operation():
    for k, x in enumerate(test):
        u = sum(x * w)
        y = 1 if u >= 0 else -1
        logging.info(f'sample {k + 1} classified as: {y}')


if __name__ == '__main__':
    perceptron()
    operation()
