#pragma once
#ifndef LAYER_H 
#define LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <jsoncpp/json/json.h>


class Layer
{
    public:
        virtual Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input) = 0;
        virtual Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate) = 0;
        virtual Json::Value toJSON() = 0;
        
        bool AsWeights();

    protected:
        Eigen::MatrixXd m_input;
        Eigen::MatrixXd m_output;
        bool m_as_weight;
        
};

#endif