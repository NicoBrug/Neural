/**
 * \file flatten_layer.cpp
 * \brief  flatten layer
 * \author Brugie Nicolas
 * \version 0.1
 *
 * Class allowing the creation of an Dense/Full flatten layer within a neural network.
 *
 */
#include "../../includes/layers/flatten_layer.h"

using namespace Eigen;
using namespace std;
using namespace Neural;

typedef Matrix<double,Dynamic,Dynamic,RowMajor> RowMajMat; 


Flatten_Layer::Flatten_Layer(){

};

MatrixXd Flatten_Layer::Forward_propagation(MatrixXd input_data){
    this->m_input = input_data;
    Map<MatrixXd> out(this->m_input.data(), 1,this->m_input.size());
    this->m_output = out;

    return this->m_output;
};

/** Performs retro propagation on the activation layer
 * 
 *  @param output_error The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @param learning_rate The step size at each iteration
 *  @return Matrix of derived activation function 
 * 
 */
MatrixXd Flatten_Layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    Map<MatrixXd> out_err(output_error.data(), this->m_input.rows(),this->m_input.cols());
    return out_err;
};

Json::Value Flatten_Layer::toJSON(){
    Json::Value json;

    json["type"] = "FlattenLayer" ;

    return json;
};