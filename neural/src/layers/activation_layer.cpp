#include "../../includes/layers/activation_layer.h"

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;
using namespace Neural;

/** Constructor of the Activation Layer
 * 
 * 
 */
Activation_layer::Activation_layer(){
    this->m_as_weight = false;
    this->p_activation = new Than();
};

Activation_layer::Activation_layer(Activation *a){
    this->m_as_weight = false;
    this->p_activation = a;
};

 Activation_layer:: ~Activation_layer(){
     delete this->p_activation;
 };

/** Performs forward propagation on the activation layer
 * 
 *  @param input_data The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @return Matrix of activation function
 * 
 */
MatrixXd Activation_layer::Forward_propagation(MatrixXd input_data){
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
MatrixXd Activation_layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    return this->p_activation->Compute_prime(this->m_input).array()*output_error.array();
};

Json::Value Activation_layer::toJSON(){
    Json::Value json;

    json["type"] = "ActivationLayer" ;
    json["ActivationFunction"] =  this->p_activation->getType();

    return json;
};