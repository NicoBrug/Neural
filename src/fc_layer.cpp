#include "../header/fc_layer.h"

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

Fc_Layer::Fc_Layer(int input_size, int output_size){
    this->m_as_weight = true;
    this->m_weights = RandomMatrix(input_size,output_size,-0.5,0.5);
    this->m_bias = RandomMatrix(1,output_size,-0.5,0.5);
};

MatrixXd Fc_Layer::Forward_propagation(MatrixXd input_data){
    this->m_input = input_data;
    this->m_output = (input_data*this->m_weights)+this->m_bias;

    return  this->m_output;
};

MatrixXd Fc_Layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    MatrixXd input_error = output_error * m_weights.transpose();
    MatrixXd weights_error = m_input.transpose()*output_error;

    this->m_weights -= learning_rate * weights_error;
    this->m_bias -= learning_rate * output_error;

    return input_error;
};

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

void Fc_Layer::SetWeights(MatrixXd weights){
    this->m_weights = weights;
}