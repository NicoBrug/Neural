#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "./includes/network.h"
#include "./includes/layers/fc_layer.h"
#include "./includes/layers/activation_layer.h"
#include "./includes/layers/flatten_layer.h"
#include "./includes/layers/conv_layer.h"
#include "./includes/activation.h"
#include "./includes/loss.h"
#include "./includes/kernel.h"
#include "./includes/core.h"
#include "./includes/loader/mnist.h"
#include "./includes/plotter/plotter.h"

#include <EigenRand/EigenRand>
#include <eigen3/Eigen/Core>


#include <QGuiApplication>

#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QAspectEngine>

#include <Qt3DRender/qrenderaspect.h>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QMaterial>

#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QTorusMesh>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DExtras/QPhongMaterial>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QSplineSeries>
#include <QtCharts/QValueAxis>
#include <QtCharts/QCategoryAxis>


using namespace Neural;

int main(int argc, char* argv[])
{
    
    
    Activation* than = new Than();
    Activation* relu = new Relu();
    Activation* softplus = new SoftPlus();
    Activation* sig = new Sigmoid();
    Activation* leakyrel = new LeakyRelu(0.2);
    Activation* softmax = new Softmax();

    Plotter p;

    

    return 0;
}