/**
 * @file activation_layer.h
 * @brief This class is the Base class for all activation types
 * @author Brugie Nicolas <nicolasbrugie@gmail.com>
 */


#pragma once
#ifndef ACTIVATION_LAYER_H 
#define ACTIVATION_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "layer.h"
#include "../activation.h"

namespace Neural
{
    class Activation_Layer : public Layer
    {

        public:
            Activation_Layer();
            Activation_Layer(Activation *a);
            ~Activation_Layer();

            virtual Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input_data);
            virtual Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate);
            virtual Json::Value toJSON();
            
            Activation *p_activation;

    };
}
#endif