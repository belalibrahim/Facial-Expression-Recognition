import pandas as pd
from sklearn.utils import shuffle

# Read the data
data = pd.read_csv('Datasets/fer2013.csv')

# Choose only happy and sad emotions
data = data.query('2<emotion<5')

# Make the emotion column have 0 or 1
data['emotion'] = data['emotion'] - 3
data[data['emotion'] == 0] = data[data['emotion'] == 0][0:6077]
data = data.dropna()
data = data.reset_index(level=0, drop=True)

# Shuffle the data
data = shuffle(data)
data = data.dropna()
data = data.reset_index(level=0, drop=True)

# Split the data into train and test
train = data[:10000]
test = data[10000:]

train = train[['emotion', 'pixels']]
test = test[['emotion', 'pixels']]

train = train.reset_index(level=0, drop=True)
test = test.reset_index(level=0, drop=True)

train.to_csv('Datasets/train.csv', index=False)
test.to_csv('Datasets/test.csv', index=False)
