import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from visual_callbacks import ConfusionMatrixPlotter


#random seed for reproducibility
seed = 7
np.random.seed(seed)

# load and arange iris dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# One-hot encode labels
Y_cat = np_utils.to_categorical(encoded_Y)

# Use sklearn to split into random Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42)

class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
plotter = ConfusionMatrixPlotter(X_val=X_test, classes=class_names, Y_val=y_test)

# create model
model = Sequential()
model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
model.add(Dense(3, init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(X_train, y_train, nb_epoch=100, batch_size=16, callbacks=[plotter])

input("press ENTER to exit")
