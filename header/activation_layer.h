#pragma once
#ifndef ACTIVATION_LAYER_H 
#define ACTIVATION_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "layer.h"

class Activation_layer : public Layer
{

    public:
        Activation_layer();

        Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input_data);
        Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate);

        Eigen::MatrixXd Tanh(Eigen::MatrixXd x);
        Eigen::MatrixXd Tanh_prime(Eigen::MatrixXd x);
    
        Eigen::MatrixXd Sigmoid(Eigen::MatrixXd x);
        Eigen::MatrixXd Sigmoid_prime(Eigen::MatrixXd x);
        
};

#endif