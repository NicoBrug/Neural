#pragma once
#ifndef FC_LAYER_H 
#define FC_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "layer.h"

class Fc_Layer : public Layer
{

    public:
        Fc_Layer(int i ,int j);
        Fc_Layer(Eigen::MatrixXd weights ,Eigen::MatrixXd bias);

        virtual Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input);
        virtual Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate);
        virtual Json::Value toJSON();

        Eigen::MatrixXd RandomMatrix(int row, int col, float min, float max);
        Eigen::MatrixXd GetWeights();
        Eigen::MatrixXd GetBias();

        void SetWeights(Eigen::MatrixXd weights);
        void SetBias(Eigen::MatrixXd weights);

    protected:
        Eigen::MatrixXd m_weights;
        Eigen::MatrixXd m_bias;
};

#endif