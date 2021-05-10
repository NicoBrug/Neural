#include "../../includes/layers/conv_layer.h"
#include "../../includes/core.h"

using namespace Eigen;
using namespace std;
using namespace Neural;



Conv_layer::Conv_layer(std::tuple<int,int,int> dimensions, 
                       std::tuple<int,int,int> filter, 
                       int stride, 
                       int padding)
{
    this->m_depth = get<2>(dimensions);
    this->m_height = get<0>(dimensions);
    this->m_width = get<1>(dimensions);
    this->m_filter_size = get<0>(filter);
    this->m_nb_filters = get<2>(filter);
    this->m_stride = stride;
    this->m_padding = padding;

    this->m_weights = Init_filters(m_filter_size,m_nb_filters);
};

MatrixXd Conv_layer::Init_filters(int dim, int nb){
    RowMajMat filters(nb,dim*dim);
    for (int i(0); i<nb; i++ ){
            MatrixXd mat = Core::RandomMatrix(dim,dim,0,1);
            Map<RowVectorXd> v1(mat.data(), mat.size());
            filters.row(i)= v1;
    }
    return filters;
};

MatrixXd Conv_layer::Forward_propagation(MatrixXd input_data){
    this->m_input = input_data;
    RowMajMat out(this->m_nb_filters,input_data.size());
    for (int i(0); i<this->m_nb_filters; i++ ){
            MatrixXd filterarray(this->m_filter_size,this->m_filter_size);
            filterarray = this->m_weights.row(i);
            Map<MatrixXd> filtermatrix(filterarray.data(), this->m_filter_size,this->m_filter_size);
            Map<MatrixXd> temp(Core::Correlate2D(input_data,filtermatrix,1,"same").data(), 1,input_data.size());
            out.row(i) = temp;
    }
    this->m_output = out;
    return this->m_output;
};


MatrixXd Conv_layer::Backward_propagation(MatrixXd output_error, float learning_rate){
    RowMajMat in_error(this->m_nb_filters,this->m_input.size());
    RowMajMat d_weights(this->m_weights.rows(),this->m_weights.cols());
    for (int i(0); i<this->m_nb_filters; i++ ){
            MatrixXd filterarray(this->m_filter_size,this->m_filter_size);
            filterarray = this->m_weights.row(i);
            Map<MatrixXd> filtermatrix(filterarray.data(), this->m_filter_size,this->m_filter_size);
            Map<MatrixXd> outerr(output_error.data(), this->m_height,this->m_width);
            Map<MatrixXd> temperr(Core::Correlate2D(outerr,filtermatrix,1,"same").data(), 1,this->m_input.size());
            
            in_error.row(i) = temperr;
            MatrixXd output_array(this->m_filter_size,this->m_filter_size);
            output_array = output_error.row(i);
            Map<MatrixXd> output_matrix(output_array.data(), this->m_filter_size,this->m_filter_size);
           
            Map<MatrixXd> d_w(Core::Correlate2D(this->m_input,output_matrix,1,"same").data(), 1,this->m_filter_size*this->m_filter_size);
            d_weights.row(i) = d_w;
    }

    this->m_weights.noalias() -= learning_rate * d_weights;
    return in_error;
};

MatrixXd Conv_layer::GetWeights(){
    return this->m_weights;
};

MatrixXd Conv_layer::GetBias(){
    return this->m_bias;
};


Json::Value Conv_layer::toJSON(){
    Json::Value json;

    json["type"] = "Conv_layer" ;

    return json;
};