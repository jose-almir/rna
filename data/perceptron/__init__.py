import pandas as pd

opts = {'header': None, 'delimiter': ' ', 'index_col': 0}

train = pd.read_table('./data/perceptron/train.txt', **opts)
test = pd.read_table('./data/perceptron/test.txt', **opts)

train.insert(0, 0, -1)
test.insert(0, 0, -1)

train = train.to_numpy()
test = test.to_numpy()
