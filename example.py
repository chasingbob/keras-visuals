from keras.models import Sequential
from keras.layers import Dense
import numpy
from visual_callbacks import AccLossPlotter

numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Instantiate AccLossPlotter to visualise training
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)

# Add your plotter to the fit function's callbacks list
model.fit(X, Y, validation_split=0.2, nb_epoch=150, batch_size=10, callbacks=[plotter])

# evaluate the model
scores = model.evaluate(X, Y)
print("\r\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
input('Press ENTER to continue...')
