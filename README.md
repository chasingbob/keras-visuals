# keras-visuals
Graphs to help you visualise the training of your Keras models.


Graph after 50 epochs
![Accuracy](/img/s1.png)


Graph after 150 epochs
![Loss](/img/s2.png)

The graphs are dynamic and will update after each epoch during the fit function.

##The code

We import the *AccLossPlotter* class from the *visual_callbacks* package.

```python
from visual_callbacks import AccLossPlotter

```

**Instantiate the plotter**
```python
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
```
* *graphs* is a list of the different graphs we would like to plot.  
* *save_graph* tells the Plotter to save a screenshot when training is finished.


**Register callback with model**

```python
model.fit(X, Y, validation_split=0.2, nb_epoch=150, batch_size=10, callbacks=[plotter])
```



