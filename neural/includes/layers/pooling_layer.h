/**
 * @file pooling_layer.h
 * @brief This class allows the creation of a Pooling layer
 * @author Brugie Nicolas <nicolasbrugie@gmail.com>
 */

#pragma once
#ifndef POOLING_LAYER_H 
#define POOLING_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <EigenRand/EigenRand>
#include "layer.h"
#include <chrono>

namespace Neural
{
    class Pooling_layer : public Layer
    {

        public:
            Pooling_layer();
            virtual Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input);
            virtual Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate);
            virtual Json::Value toJSON();


        protected:

    };
}
#endif