# keras-visuals
Graphs to help you visualise the training of your Keras models.


![Accuracy](/img/s1.png)


![Loss](/img/s2.png)

The graphs are dynamic and will update after each epoch during the fit function.

##The code

We import the AccLoss Plotter class from the visual_callbacks pacakge

```python
from visual_callbacks import AccLossPlotter

```

Instantiate the plotter
```python
plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True)
```
*graphs* is a list of the different graphs we would like to plot. Available options: Accuracy (acc), Loss (loss) 
*save_graph* tells the Plotter to save a screenshot on the train_end event

Register callback with model

```python
model.fit(X, Y, validation_split=0.2, nb_epoch=150, batch_size=10, callbacks=[plotter])
```



