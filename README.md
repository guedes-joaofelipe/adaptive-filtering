# Adaptive Filtering

Repository containing a Python implemetation of the Matlab [Adaptive Filtering toolbox](http://www.mathworks.com/matlabcentral/fileexchange/3582-adaptive-filtering). 

All classes can be found in the `adaptive_filtering` folder. A notebook containing solutions for exercises in [Adaptive Filtering: Algorithms and Practical Implementation](https://www.springer.com/gp/book/9781461441052) can be found in the `Notebooks` folder. 

## Overview 

In many real world scenarios, we are interested in discovering the coefficients that rule a transfer function of a given filtering system. Usually, their can be determined by specifications of frequency operations such as pass-band ranges and gain. 

However, dynamic environments require that such coefficients become adaptive to changes in their frequency response. In other words, these coefficients need to be adapt to incoming data so as to achieve required specifications. 

In this sense, adaptive filters come as a trainable solution to a number of time-series related problems. 

## LMS-based Algorithms: 

The following LMS-based algorithms have been implemented:

- LMS (Least Mean Square)
- NLMS (Normalized LMS)
- LMSNewton
- SignData
- SignError
- DualSign
- AffineProjection
- TransformDomain using DCT and DFT

## Examples 

To run examples, check the `Examples` folder. The scripts do not need any arguments through the command line and the outputs are stored by default in the `Examples/Outputs` folder (such filepath can be changed in each script though). 

Take for instance a system identifications where the standard LMS algorithm is applied: 

<img src="./Examples/schematic.png" style="float: left; margin-right: 10px;" />

The algorithm has to identify the following filter weights' vector (which is also its Wiener Solution): 

**w_o** = [0.32+0.21j, -0.3+0.7j, 0.5-0.8j, 0.2+0.5j]^T

By running the example with $K = 100$ iterations, *learning rate* = 0.1 and taking the average of 25 ensembles, the following learning curves are provided:

<img src="./Examples/Outputs/LMS_coef_evolution.jpg" style="float: left; margin-right: 10px;" />

<img src="./Examples/Outputs/LMS_learning_curve.jpg" style="float: left; margin-right: 6px;" />


