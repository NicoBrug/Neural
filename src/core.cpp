#include "../includes/core.h"

using namespace Eigen;
using namespace std;

MatrixXd Core::RandomMatrix(int rows, int cols, float min, float max){
    /* double range= max-min;
    
    MatrixXd m = MatrixXd::Random(rows,cols); 
    m = (m + MatrixXd::Constant(rows,cols,1.))*range/2.; 
    m = (m + MatrixXd::Constant(rows,cols,min));
    
    */

    MatrixXd m{rows,cols};

    Rand::Vmt19937_64 urng{ 42 };

    Rand::balancedLike(m, urng);
 
    m = Rand::balancedLike(m, urng);

    //MatrixXd M = Map<Matrix<double,Dynamic,Dynamic,RowMajor> >(m.data(), rows, cols);
    return m;
};
