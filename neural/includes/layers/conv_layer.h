/**
 * @file conv_layer.h
 * @brief This class allows the creation of a convolution layer
 * @author Brugie Nicolas <nicolasbrugie@gmail.com>
 */

#pragma once
#ifndef CONV_LAYER_H 
#define CONV_LAYER_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <EigenRand/EigenRand>
#include "layer.h"
#include <chrono>
using namespace Eigen;

typedef Matrix<double,Dynamic,Dynamic,RowMajor> RowMajMat; 

namespace Neural
{
    class Conv_Layer : public Layer
    {

        public:
            Conv_Layer(std::tuple<int,int,int> dimensions, 
                    std::tuple<int,int,int> filter, 
                    int stride, 
                    int padding);

            virtual Eigen::MatrixXd Forward_propagation(Eigen::MatrixXd input);
            virtual Eigen::MatrixXd Backward_propagation(Eigen::MatrixXd output_error, float learning_rate);
            virtual Json::Value toJSON();
            Eigen::MatrixXd Init_filters(int dim, int nb);
            Eigen::MatrixXd GetWeights();
            Eigen::MatrixXd GetBias();
            
        protected:
            int m_depth;
            int m_height;
            int m_width;
            int m_filter_size;
            int m_nb_filters;
            int m_stride;
            int m_padding;

            RowMajMat m_weights;
            RowMajMat m_bias;

            RowMajMat m_filter;
    };
}
#endif