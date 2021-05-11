/**
 * \file activation.cpp
 * \brief  layer for activation 
 * \author Brugie Nicolas
 * \version 0.1
 *
 * Class allowing the creation of an activation layer within a neural network.
 *
 */

#include "../../includes/layers/activation_layer.h"

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;
using namespace Neural;

/** Constructor of the Activation Layer
 * 
 * 
 */
Activation_Layer::Activation_Layer(){
    this->m_as_weight = false;
    this->p_activation = new Than();
};

Activation_Layer::Activation_Layer(Activation *a){
    this->m_as_weight = false;
    this->p_activation = a;
};

 Activation_Layer:: ~Activation_Layer(){
     delete this->p_activation;
 };

/** Performs forward propagation on the activation layer
 * 
 *  @param input_data The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @return Matrix of activation function
 * 
 */
MatrixXd Activation_Layer::Forward_propagation(MatrixXd input_data){
    this->m_input = input_data;
    return this->p_activation->Compute(input_data);
};

/** Performs retro propagation on the activation layer
 * 
 *  @param output_error The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @param learning_rate The step size at each iteration
 *  @return Matrix of derived activation function 
 * 
 */
MatrixXd Activation_Layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    return this->p_activation->ComputePrime(this->m_input).array()*output_error.array();
};

Json::Value Activation_Layer::toJSON(){
    Json::Value json;

    json["type"] = "ActivationLayer" ;
    json["ActivationFunction"] =  this->p_activation->getType();

    return json;
};