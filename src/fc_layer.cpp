#include "../includes/fc_layer.h"
#include "../includes/core.h"
#include "../includes/kernel.h"

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

/** Constructor of the Full connected Layer, Generate Weight & Bias random Matrix 
 * 
 *  @param input_size size of rows  
 *  @param output_size size of cols  
 *
 */
Fc_Layer::Fc_Layer(int input_size, int output_size){
    this->m_as_weight = true;
    this->m_weights = Core::RandomMatrix(input_size,output_size,-0.5,0.5);
    this->m_bias = Core::RandomMatrix(1,output_size,-0.5,0.5);

};

/** Constructor of the Full connected Layer, Fill Weight & Bias with preexistant Matrix 
 * 
 *  @param weights Matrix of weight  
 *  @param bias size of cols  
 *
 */
Fc_Layer::Fc_Layer(MatrixXd weights, MatrixXd bias){
    this->m_as_weight = true;
    this->m_weights = weights;
    this->m_bias = bias;
};

/** Performs forward propagation on the current layer
 * 
 *  @param input_data The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @return Output Matrix of forward propagation results 
 * 
 */
MatrixXd Fc_Layer::Forward_propagation(MatrixXd input_data){
    this->m_input = input_data;
    this->m_output = (input_data*this->m_weights)+this->m_bias;

    return  this->m_output;
};

/** Performs retro propagation on current layer
 * 
 *  @param output_error The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 *  @param learning_rate The step size at each iteration
 *  @return Matrix of input Layer error 
 *  /!\ -> Matrix stored row or col 
 */
MatrixXd Fc_Layer::Backward_propagation(MatrixXd output_error, float learning_rate){


    //GPU
    //double* input_error_array = Kernel::dot(output_error.data(), m_weights.transpose().data(),output_error.rows(),output_error.cols(),m_weights.rows()); 
    //double* weights_error_array = Kernel::dot(m_input.transpose().data(), output_error.data(),m_input.cols(),m_input.rows(),output_error.cols()); 
    
    //CPU
    MatrixXd input_error = output_error*m_weights.transpose();
    MatrixXd weights_error = m_input.transpose()*output_error;

    //MatrixXd input_error = Map<Matrix<double,Dynamic,Dynamic> >(input_error_array,output_error.rows(),m_weights.rows());
    //MatrixXd weights_error = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(weights_error_array,m_input.cols(),output_error.cols());

    this->m_weights.noalias() -= learning_rate * weights_error;
    this->m_bias.noalias() -= learning_rate * output_error;
    return input_error;
};

Json::Value Fc_Layer::toJSON(){
    Json::Value json;

    Json::Value weights(Json::arrayValue);
    Json::Value bias(Json::arrayValue);

    for (int i(0); i<this->m_weights.rows();i++){
        Json::Value rows(Json::arrayValue);

        for (int j(0); j<this->m_weights.cols();j++){
            rows.append(Json::Value(this->m_weights(i,j)));
        }
        weights.append(rows);

    }

    for (int i(0); i<this->m_bias.rows();i++){
        Json::Value rows(Json::arrayValue);

        for (int j(0); j<this->m_bias.cols();j++){
                rows.append(Json::Value(this->m_bias(i,j)));
        }
        bias.append(rows);
    }

    json["weights"] = weights ; 
    json["type"] = "FcLayer" ;
    json["bias"] = bias;

    return json;
};

MatrixXd Fc_Layer::GetWeights(){
    return this->m_weights;
};

MatrixXd Fc_Layer::GetBias(){
    return this->m_bias;
};

void Fc_Layer::SetWeights(MatrixXd weights){
    m_weights = weights;
}

void Fc_Layer::SetBias(MatrixXd bias){
    m_bias = bias;
}