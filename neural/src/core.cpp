#include "../includes/core.h"

using namespace Eigen;
using namespace std;
using namespace Neural;

MatrixXd Core::RandomMatrix(int rows, int cols, float min, float max){
    MatrixXd m{rows,cols};

    auto seed = std::random_device{}();
    Rand::Vmt19937_64 urng{ seed };
    Rand::balancedLike(m, urng);
    m = Rand::balancedLike(m, urng);
    
    MatrixXd M = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(m.data(), rows, cols);
    return m;
};

// VALID ->  [(n x n) image] * [(f x f) filter] —> [(n – f + 1) x (n – f + 1) image]
// SAME  ->   [(n + 2p) x (n + 2p) image] * [(f x f) filter] —> [(n x n) image]  -> p = (f-1)/2
MatrixXd Core::Correlate2D(MatrixXd input, MatrixXd filter, int stride, string padding){
    MatrixXd pad;
    int sizeFilter = filter.rows();

    if (padding == "same"){
        MatrixXd res(input.rows(),input.cols());
        pad = Core::Padding(input, (filter.rows()-1)/2);
        int _i(0);
        for (int i(0);i<input.rows(); i+=stride){
            int _j(0);
            for (int j(0); j<input.cols(); j+=stride){
                MatrixXd temp = pad.block(i,j,sizeFilter,sizeFilter);
                double acc = temp.cwiseProduct(filter).sum();
                res(_i,_j) = acc;
                _j++;
            }
            _i++;
        }
        return res;
    }

    MatrixXd res(input.rows()-filter.rows()+1,input.cols()-filter.cols()+1);
    pad = input;
    int _i(0);
    for (int i(stride);i<input.rows()-stride; i+=stride){
        int _j(0);
        for (int j(stride); j<input.cols()-stride; j+=stride){
            MatrixXd temp = pad.block(i-1,j-1,sizeFilter,sizeFilter);
            double acc = temp.cwiseProduct(filter).sum();
            res(_i,_j)=acc;
            _j++;
        }
        _i++;
    }
    return res;
};

MatrixXd Core::Padding(MatrixXd m, int p){
    MatrixXd a = MatrixXd::Zero(p*2+m.rows(),p*2+m.cols());
    a.block(p,p,m.rows(),m.cols())=m;
    return a;
};

void Core::PrintArray(double *array){
    for (int i = 0; i <= sizeof(array); i++) {
        cout << array[i] << "  ";
    }
    cout << "\n" << endl;
}
