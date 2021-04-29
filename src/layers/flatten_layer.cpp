#include "../../includes/layers/flatten_layer.h"

using namespace Eigen;
using namespace std;
typedef Matrix<double,Dynamic,Dynamic,RowMajor> RowMajMat; 


Flatten_layer::Flatten_layer(){

};

MatrixXd Flatten_layer::Forward_propagation(MatrixXd input_data){
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
MatrixXd Flatten_layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    Map<MatrixXd> out_err(output_error.data(), this->m_input.rows(),this->m_input.cols());
    return out_err;
};

Json::Value Flatten_layer::toJSON(){
    Json::Value json;

    json["type"] = "FlattenLayer" ;

    return json;
};