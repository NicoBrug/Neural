Neural                         {#mainpage}
============

## Description
Neural is a library developed in C++ allowing the realization of artificial neural networks. It is intended to be simple to use and easy to install (Just create the Docker and start to code!). If you encounter any problem, don't hesitate to let me know. 

It is built on the basis of eigen which allows to facilitate the manipulation of data structures, and to have various functions of manipulation of matrices or lists. Moreover, Neural has natively a multi-threading function which allows you to easily add the number of threads you want to allocate to optimize the performances you need.

Moreover, some operations are already supported on CUDA like matrix multiplication. Other functions like cross-correlation and convolution are currently under development. For more informations about this subject, thanks to refer to Cuda Support.

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
Adam                | \emoji soon   |
Adagrad             | \emoji x |
Momentum            | \emoji x |

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

