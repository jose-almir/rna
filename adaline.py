import logging

import numpy as np

from data.perceptron import train, test

logging.basicConfig(
    format='[%(asctime)s](%(levelname)s) %(message)s',
    level=logging.DEBUG,
    datefmt='%H:%M:%S'
)

w = np.random.rand(4)
n = .0025
e = 1e-6
epochs = np.inf

logging.info('adaline inputs configured')


def __eqm():
    eqm = 0

    for sample in train:
        u = sum(sample[:-1] * w)
        d = sample[-1]
        eqm += (d - u) ** 2

    return eqm / len(train)


def adaline():
    logging.debug(f'w={w}')

    epoch = 0

    while True:
        eqm_prev = __eqm()

        for sample in train:
            u = sum(sample[:-1] * w)
            d = sample[-1]

            for j in range(len(w)):
                w[j] += n * (d - u) * sample[j]

        epoch += 1

        eqm_curr = __eqm()

        if abs(eqm_curr - eqm_prev) <= e:
            logging.info(f"|EQM - EQM|={abs(eqm_curr - eqm_prev)}")
            logging.debug(f'w={w}')
            break


def operation():
    for k, x in enumerate(test):
        u = sum(x * w)
        y = 1 if u >= 0 else -1
        logging.info(f'sample {k + 1} classified as: {y}')


if __name__ == '__main__':
    adaline()
    operation()
