#include "../../includes/plotter/plotter.h"

using namespace Neural;
using namespace Eigen;
using namespace std;

Plotter::Plotter(){

};


void Plotter::Plot(Activation *func){
   vector<double> data;

   sciplot::Plot plot;
   
   int nb = 100;
   for (int i(-nb); i<=nb; i++){
       data.push_back(i*0.1);
   }

   MatrixXd d(1,nb*2);
   double* ptr = &data[0];
   Map<RowVectorXd> v(ptr,nb*2);
   d.row(0) = v;

  
    MatrixXd res = func->Compute(d);

    std::vector<double> v3(&res(0), res.data()+res.cols()*res.rows());

    
    plot.xlabel("x");
    plot.ylabel("y");

    plot.xrange(-2, 2);
    plot.yrange(-2.0, 2.0);
    
    plot.legend().atOutsideTop();


    plot.drawCurve(data, v3).label(func->getType());
    plot.size(400, 400 );
    plot.xtics().logscale(2);    

    plot.show();

    /* sciplot::Vec x = { 1, 2, 3 };
    sciplot::Vec y = { 4, 5, 6 };

    sciplot::Plot plot1;
    
    plot1.drawPoints(x, y).pointType(0);

    plot1.show(); */
  
}