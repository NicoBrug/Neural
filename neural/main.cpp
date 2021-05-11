#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "./includes/network.h"
#include "./includes/layers/fc_layer.h"
#include "./includes/layers/activation_layer.h"
#include "./includes/layers/flatten_layer.h"
#include "./includes/layers/conv_layer.h"
#include "./includes/activation.h"
#include "./includes/loss.h"
#include "./includes/kernel.h"
#include "./includes/core.h"
#include "./includes/loader/mnist.h"
#include "./includes/plotter/plotter.h"

#include <EigenRand/EigenRand>
#include <eigen3/Eigen/Core>




using namespace Neural;
using namespace std;
using namespace Eigen;

int main()
{
    mnist train("../dataset/MNIST/train-images-idx3-ubyte",
		     "../dataset/MNIST/train-labels-idx1-ubyte", 10000);

    mnist test("../dataset/MNIST/t10k-images-idx3-ubyte",
		     "../dataset/MNIST/t10k-labels-idx1-ubyte", 30);

    
    Network net;

    Plotter p;
    Activation* elu = new Elu(1);
    p.Plot(elu);

    return 0;
}