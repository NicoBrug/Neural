#include "../header/fc_layer.h"

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

/** Constructor of the Full connected Layer
 * 
 * 
 */
Fc_Layer::Fc_Layer(int input_size, int output_size){
    this->m_as_weight = true;
    this->m_weights = RandomMatrix(input_size,output_size,-0.5,0.5);
    this->m_bias = RandomMatrix(1,output_size,-0.5,0.5);
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
 * 
 */
MatrixXd Fc_Layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    MatrixXd input_error = output_error * m_weights.transpose();
    MatrixXd weights_error = m_input.transpose()*output_error;

    this->m_weights -= learning_rate * weights_error;
    this->m_bias -= learning_rate * output_error;

    return input_error;
};

/** Create customized random matrix
 * 
 *  @param rows Number of row of Matrix 
 *  @param cols Number of column of Matrix 
 *  @param min Min value for random generation
 *  @param max Max value for random generation
 *  @return Random of n row et p col of value between min and max;
 * 
 */
MatrixXd Fc_Layer::RandomMatrix(int rows, int cols, float min, float max){
    double range= max-min;
    
    MatrixXd m = MatrixXd::Random(rows,cols); // n*p Matrix filled with random numbers between (-1,1)
    m = (m + MatrixXd::Constant(rows,cols,1.))*range/2.; // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    m = (m + MatrixXd::Constant(rows,cols,min));

    return m;
};


MatrixXd Fc_Layer::GetWeights(){
    return this->m_weights;
};

MatrixXd Fc_Layer::GetBias(){
    return this->m_bias;
};

void Fc_Layer::SetWeights(MatrixXd weights){
    this->m_weights = weights;
}

void Fc_Layer::SetBias(MatrixXd bias){
    this->m_bias = bias;
}