#pragma once
#ifndef ACTIVATION_LAYER_H 
#define ACTIVATION_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "layer.h"
#include "../activation.h"

namespace Neural
{
    class Activation_layer : public Layer
    {

        public:
            Activation_layer();
            Activation_layer(Activation *a);
            ~Activation_layer();

            virtual Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input_data);
            virtual Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate);
            virtual Json::Value toJSON();
            
            Activation *p_activation;

    };
}
#endif