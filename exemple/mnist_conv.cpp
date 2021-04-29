#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "../includes/network.h"
#include "../includes/fc_layer.h"
#include "../includes/activation_layer.h"
#include "../includes/flatten_layer.h"
#include "../includes/activation.h"
#include "../includes/loss.h"
#include "../includes/kernel.h"
#include "../includes/core.h"
#include "../includes/loader/mnist.h"
#include "../includes/conv_layer.h"

#include <EigenRand/EigenRand>
#include <eigen3/Eigen/Core>

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;



int main() {

    mnist train("../dataset/MNIST/train-images-idx3-ubyte",
		     "../dataset/MNIST/train-labels-idx1-ubyte", 1000);

    mnist test("../dataset/MNIST/t10k-images-idx3-ubyte",
		     "../dataset/MNIST/t10k-labels-idx1-ubyte", 30);

    int rows  = train.rows();
    int cols  = train.cols();
    
    int rowstest = test.rows();
    int colstest = test.cols();
    
    //Prepare data : train -> array of pixel matrix
    //Prepare data : label -> array[10] for output result 
    MatrixXd mnist_matrix(train.size(),rows*cols);
    MatrixXd mnist_label(train.size(),10);

    MatrixXd mnist_test(test.size(),rows*cols);
    MatrixXd mnist_test_label(test.size(),10);

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

    for (int i(0); i < test.size(); i++){
        std::vector<double> image = test.images(i);
        double size = cols * rows;
        mnist_test.row(i) = Map<Matrix<double,1,784> >(image.data());

        MatrixXd label(1,10);

        for (int j(0); j<10; j++){
            if (test.labels(i) == j){
                label(0,j) = 1.0;
            }
            else{
                label(0,j) = 0.0;
            }
        }
        mnist_test_label.row(i) = label;
    }

    std::tuple<int, int, int> dimensions1 = std::make_tuple(28,28,1);
    std::tuple<int, int, int> filter1 = std::make_tuple(3,3,1);

    std::tuple<int, int, int> dimensions2 = std::make_tuple(8,8,1);
    std::tuple<int, int, int> filter2 = std::make_tuple(3,3,1);

    std::tuple<int, int, int> dimensions3 = std::make_tuple(6,6,1);
    std::tuple<int, int, int> filter3 = std::make_tuple(3,3,2);

    Network net; 
    Activation* than = new Than();
    Loss* mse = new Mse();
    net.Use(mse);

    Conv_layer* c1 = new Conv_layer(dimensions1,filter1,1,1);
    Activation_layer* acl1 = new Activation_layer(than);
    Flatten_layer* fl = new Flatten_layer();
    Fc_Layer* fc1 = new Fc_Layer(784,100);
    Activation_layer* acl2 = new Activation_layer(than);
    Fc_Layer* fc2 = new Fc_Layer(100,10);
    Activation_layer* acl3 = new Activation_layer(than);


    net.Add(c1);
    net.Add(acl1);
    net.Add(fl);
    net.Add(fc1);
    net.Add(acl2);
    net.Add(fc2);
    net.Add(acl3);

    net.Fit(mnist_matrix,mnist_label,100,0.1,1);
    net.Predict(mnist_test);

    cout << "result true \n" << mnist_test_label << endl;
    return 0;
}