Neural                         {#mainpage}
============

## Description
Neural is a framework developed in C++ allowing the realization of artificial neural networks. This framework includes a library for the realization of neural networks, a powerful linear algebra library (Eigen), a docker image to facilitate the implementation of CUDA and Open GL, and a graphical visualization library for displaying and debugging the neural network. This framework is easy to use, and to install. If you encounter any problems, don't hesitate to report them. 

Some operations are already supported on CUDA like matrix multiplication. Other functions like cross-correlation and convolution are currently under development. For more informations about this subject, thanks to refer to Cuda Support.


### Linear Algebric  
The Neural API is built on the basis of Eigen. [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is a numerical analysis library in C++ composed of template headers. 

### Neural networks Components

\emoji heavy_check_mark : Developed and tested

\emoji soon  : In development

\emoji x : Not Implemented (but planned)

Research : In the research phase 

<table>
<tr><th>Optimizer</th><th>Layers</th><th>Activation</th><th>Loss</th></tr>
<tr><td>

Type          | Status                        |
------------------- | ------                  |
SGD                 | \emoji heavy_check_mark |
Momentum            | \emoji soon             |
Nesterov            | \emoji soon             |
Adam                | \emoji x                |
Adagrad             | \emoji x                |

</td><td>

Type                   | Status                   | 
----------------------- | ------                  | 
Full connected (dense)  | \emoji heavy_check_mark | 
Activation              | \emoji heavy_check_mark | 
Convolution             | \emoji heavy_check_mark | 
Flatten                 | \emoji heavy_check_mark |   
Pooling                 | \emoji soon             |    
Dropout                 | \emoji soon             |    
Associative             | Research                |    

</td><td>

Type          | Status                        |
------------------- | ------                  |
Hyperbolic tangent  | \emoji heavy_check_mark |
Sigmoid             | \emoji heavy_check_mark |
Relu                | \emoji heavy_check_mark |
LeakyRelu           | \emoji heavy_check_mark |
SoftPlus            | \emoji heavy_check_mark |
Softmax             | \emoji x                |

</td><td>

Type      |Status                           |
------------------ |------                  |
Mean squared error |\emoji heavy_check_mark |
Cross entropy      |\emoji x                |
Mean Absolute error|\emoji soon             |
Mean Bias Error    |\emoji soon             |
Hinge Loss         |\emoji x                | 


</td></tr> 

</table>

### GUI
Neural allows the graphical display of results and functions usable from the API. 

### CUDA support
In order to access the power of the GPU, we use a nvidia-cuda container to configure and access the GPU easily. This containerization technology allows to have access to a runtime library and utilities to automatically configure containers to leverage NVIDIA GPUs.
The GPU contextualization, allows to facilitate the application deployment, to isolate the different devices, to allow better performances in multi-GPU.

\image html images/nvidia-docker.png width=500px

