Neural                         {#mainpage}
============

## Description
Neural is a library developed in C++ allowing the realization of artificial neural networks. It is intended to be simple to use and easy to install (Just create the Docker and start to code!). If you encounter any problem, don't hesitate to let me know. 

It is built on the basis of eigen which allows to facilitate the manipulation of data structures, and to have various functions of manipulation of matrices or lists. Moreover, eigen has natively a multi-threading function which allows you to easily add the number of threads you want to allocate to optimize the performances you need.



### Components

<table>
<tr><th>Layers</th><th>Activation</th><th>Loss</th></tr>
<tr><td>

Type                   | Status                  | 
----------------------- | ------                  | 
Full connected (dense)  | \emoji heavy_check_mark | 
Activation              | \emoji heavy_check_mark | 
Convolution             | \emoji heavy_check_mark | 
Flatten                 | \emoji heavy_check_mark |   
Pooling                 | \emoji soon             |    
Associative             | Research                |    

</td><td>

Type          | Status                  |
------------------- | ------                  |
Hyperbolic tangent  | \emoji heavy_check_mark |
Sigmoid             | \emoji heavy_check_mark |
Relu                | \emoji heavy_check_mark |
SoftPlus            | \emoji heavy_check_mark |
Softmax             | \emoji x                |

</td><td>

Type      |Status                  |
------------------ |------                  |
Mean squared error |\emoji heavy_check_mark |
Cross entropy      |\emoji x                |
Mean Absolute error|\emoji soon             |
Mean Bias Error    |\emoji soon             |
Hinge Loss         |\emoji x                | 
</td></tr> 

</table>

