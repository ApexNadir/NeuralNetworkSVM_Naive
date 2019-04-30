# NeuralNetworkSVM_Naive
This program is a naive implementation of Multi layer Perceptrons with a sigmoid activation function. 
The neurons are implemented as objects. A better implementation would use matrices to represent network layers.

The data set is constructed using two attributes, X and Y as predictive attributes, and two goal-field attributes RED and GREEN. 
The goal field attributes are probabilities that the classification is RED or GREEN given the predictive attributes X and Y.

Controls
Left click and Right click adds a training example at the position the mouse was clicked, using X and Y as two predictive attributes.
C clears the data set and generates a new NeuralNetwork model.
R adds 20 random data points.
P pauses the network, when paused new data points can still be added, the network won't train on those until P is pressed again to unpause.
