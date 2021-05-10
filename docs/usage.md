\page ApiPage Neural API
[TOC]


\section Sec1  Core

\section Sec2 Loss

Loss functions are used to determine the error (aka “the loss”) between the output of our algorithms and the given target value.  In layman’s terms, the loss function expresses how far off the mark our computed output is.

\subsection SubLoss1 Mean Squared error

Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. MSE is a risk function, corresponding to the expected value of the squared error loss. 
```
Loss* mse = new Mse(); 
mse->Compute(Matrix x, Matrix y);
mse->ComputePrime(Matrix x, Matrix y);
```
\subsection SubLoss2 Cross Entropy

Cross-entropy is a measure from the field of information theory, building upon entropy and generally calculating the difference between two probability distributions

```
Loss* cre = new Cross_entropy(); 
cre->Compute(Matrix x, Matrix y);
cre->ComputePrime(Matrix x, Matrix y);
```

\section Sec3 Activation
An Activation function are use to get the output of node. It is also known as Transfer Function.

\subsection SubActi1 Than

This function is defined as the ratio between the hyperbolic sine and the cosine functions 

\image html images/tanhPlot.png width=300px

```
Activation* than = new Than();
than->Compute(MatrixXd x);
than->Compute_Prime(MatrixXd x);
```

\subsection SubActi2 Sigmoid

The following sigmoid activation function converts the weighted sum to a value between 0 and 1.

\image html images/sigmoidPlot.png width=300px

```
Activation* sig = new Sigmoid();
sig->Compute(MatrixXd x);
sig->Compute_Prime(MatrixXd x);
```


\subsection SubActi3 SoftPlus

Soft Plus function 

\image html images/softPlusPlot.png width=300px

```
Activation* sp = new SoftPlus();
sp->Compute(MatrixXd x);
sp->Compute_Prime(MatrixXd x);
```

\subsection SubActi4 ReLU

This function allows us to perform a filter on our data. It lets the positive values (x > 0) pass in the following layers of the neural network. It is used almost everywhere but not in the final layer, it is used in the intermediate layers.

\image html images/ReluPlot.png width=300px

```
Activation* relu = new Relu();
relu->Compute(MatrixXd x);
relu->Compute_Prime(MatrixXd x);
```

\subsection SubActi5 LeakyReLU

Leaky ReLUs allow a small, positive gradient when the unit is not active

\image html images/leakyReluPlot.png width=300px

```
Activation* Lrelu = new LeakyRelu(alpha);
Lrelu->Compute(MatrixXd x);
Lrelu->Compute_Prime(MatrixXd x);
```


\subsection SubActi6 SoftPlus

\section Sec4 Layers

\subsection SubLayers Convolution
Apply a Convolution on the inputs. The Convolution layer takes a tuple of 3 integers for the dimensions of the input and the kernel, and an integer for the stride and padding. 
```
Conv_layer( dimensions<int,int,int>, filter<int,int,int>, stride=1,padding=1 );
```
\subsection SubLayers1 Flatten

\subsection SubLayers2 Full Connected


