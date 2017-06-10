# keras-visuals
Graphs to help you visualise the training of your Keras models.


## Accuracy & loss graph

Graph after 50 epochs

<img src="/img/s1.png" width="400"/>


Graph after 150 epochs

<img src="/img/s2.png" width="400"/>

The graphs are dynamic and will automatically update and scale: after each epoch during the fit function.

#### The code

**Import AccLossPlotter**

```python
from visual_callbacks import AccLossPlotter

```

**Instantiate the plotter**
```python
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
```
* *graphs* is a list of the different graphs we would like to plot. Available ('acc', 'loss')
* *save_graph* tells the Plotter to save a screenshot when training is finished.


**Register callback with model**

```python
model.fit(X, Y, validation_split=0.2, nb_epoch=150, batch_size=10, callbacks=[plotter])
```

## Confusion Matrix

After 50 epochs

<img src="/img/c1.png" width="400"/>


After 100 epochs

<img src="/img/_c2.png" width="400"/>

It is clear from the confusion matrix that your model is confusing *iris-versicolor* for *iris-virginica*. Directed insight like this is a valuable tool for finding problem areas and improving your model.


#### The code

We import the *ConfusionMatrixPlotter* class from the *visual_callbacks* package.


```python
from visual_callbacks import ConfusionMatrixPlotter

```

**Instantiate the plotter**
```python
plotter = ConfusionMatrixPlotter(X_val=X_test, classes=class_names, Y_val=y_test)
```

* *X_val* is a list of input values, this should typically be a seperate test set
* *Y_val* is the list of output values for your X_val input set
* *classes* is a list of class names

**Register callback with model**

```python
model.fit(X_train, y_train, nb_epoch=100, batch_size=16, callbacks=[plotter])
```

**What is next**

* Visualising Neural Network Layer Activation 
* t-SNE visualisation

**Collaboration**

Feel free to get in touch or send me a *Pull Request*

* e: dries.cronje@outlook.com
* t: @dries139





