#include "../header/activation_layer.h"

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

/** Constructor of the Activation Layer
 * 
 * 
 */
Activation_layer::Activation_layer(){
    this->m_as_weight = false;
    
};

/** Performs forward propagation on the activation layer
 * 
 *  @param input_data The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @return Matrix of activation function
 * 
 */
MatrixXd Activation_layer::Forward_propagation(MatrixXd input_data){
    this->m_input = input_data;
    return Tanh(input_data);
};

/** Performs retro propagation on the activation layer
 * 
 *  @param output_error The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @param learning_rate The step size at each iteration
 *  @return Matrix of derived activation function 
 * 
 */
MatrixXd Activation_layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    return Tanh_prime(this->m_input).array()*output_error.array();
};

//Hyperbolic tangant
MatrixXd Activation_layer::Tanh(MatrixXd x){
    return x.array().tanh();
}

//Hyperbolic tangant prime
MatrixXd Activation_layer::Tanh_prime(MatrixXd x){
    return 1-x.array().tanh().pow(2);
}

//Sigmoid /!\ Not implemented
MatrixXd Activation_layer::Sigmoid(MatrixXd x){
    return x;  // 1/(1+exp(-x))
}

//Sigmoid prime /!\ Not implemented
MatrixXd Activation_layer::Sigmoid_prime(MatrixXd x){
    return x;
}