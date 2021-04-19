#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../header/network.h"
#include "../header/fc_layer.h"
#include "../header/activation_layer.h"
#include "../header/activation.h"

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;

int main() {
    
    //Load data
    mnist train("../dataset/MNIST/train-images-idx3-ubyte",
		     "../dataset/MNIST/train-labels-idx1-ubyte", 200);
    
    int rows  = train.rows();
    int cols  = train.cols();

    //Prepare data : train -> array of pixel matrix
    //Prepare data : label -> array[10] for output result 
    MatrixXd mnist_matrix(train.size(),rows*cols);
    MatrixXd mnist_label(train.size(),10);

    for (int i(0); i < train.size(); i++){
        std::vector<double> image = train.images(i);
        double size = cols * rows;
        mnist_matrix.row(i) = Map<Matrix<double,1,784> >(image.data());

        MatrixXd label(1,10);

        for (int j(0); j<10; j++){
            if (train.labels(i) == j){
                label(0,j) = 1.0;
            }
            else{
                label(0,j) = 0.0;
            }
        }
        mnist_label.row(i) = label;

    }

    //Set network
    Network net;

    Activation* than = new Than();

    Loss* mse = new Mse();
    net.Use(mse);

    Fc_Layer* fcl1 = new Fc_Layer(784,200);
    Activation_layer* acl1 = new Activation_layer(than);
    Fc_Layer* fcl2 = new Fc_Layer(200,200);
    Activation_layer* acl2 = new Activation_layer(than);
    Fc_Layer* fcl3 = new Fc_Layer(200,10);
    Activation_layer* acl3 = new Activation_layer(than);

    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);
    net.Add(fcl3);
    net.Add(acl3);
    
    //Fit network
    net.Fit(mnist_matrix,mnist_label,100,0.01);
    

    return 0;

}