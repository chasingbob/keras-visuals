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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]



# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
Y_cat = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42)


# create model
model = Sequential()
model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
model.add(Dense(3, init='normal', activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=50, batch_size=16)

result = model.evaluate(X_test, y_test)


pred = model.predict(X_test)
_pred = np.zeros(3*len(X_test)).reshape((len(X_test), 3))
print(pred)

max_pred = np.argmax(pred, axis=1)
max_y = np.argmax(y_test, axis=1)
cnf_mat = confusion_matrix(max_y, max_pred)
print(cnf_mat)
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

conf_plotter = ConfusionMatrixPlotter()
conf_plotter.update(cnf_mat, class_names)

#plt.figure()
#plot_confusion_matrix(cnf_mat, classes=class_names,title='Confusion matrix', normalize=True)
#plt.show()
input("press ENTER to exit")
