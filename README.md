# Neural circuit model for time perception

This code is an implementation of the circuit as published by Seth W. Egger, Nhat M. Le, and Mehrdad Jazayeri. "A neural circuit model for human sensorimotor timing." Nature Communications (2020).

The circuit is used to simulate time interval reproduction. 

## Circuit Implementation
In the code/onetwogo folder, code for the circuit implementation can be found. 

The code was designed in a modular way, such that multiple types of experimental procedures with shared functionality access the same basic circuit, which is implemented in `BaseSimulation`. The implementation of the basic circuit can be reused for all epochs of the experiment. 

The interval reproduction experiment, as described in this work, is implemented in experiment `simulation.py`. Parallel simulations of one trial (one delay, measurement, and reproduction epoch) are implemented in parallel `simulation.py`. 
Both experiment simulation.py and parallel `simulation.py` contain a simulate function that accesses the base
simulation (`BaseSimulation`) with the implementation of the basic circuit. 

Different experiments can have different result types, all of which can be found in `result.py`.
Analysis of the results is performed in the same file.

Visualization of results is implemented in `plot.py`.

## Simulation results
The Ipython notebooks in the code folder explore different model regimes and simulation results. 

### Thesis
A concise overview of the model is illustrated in report/poster.
In the thesis PDF a detailed description of the model and the experiment simulation results can be found.
