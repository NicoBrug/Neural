#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <eigen3/Eigen/Core>

namespace Kernel
{
    double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2);
    double * dot(const double *m1, const double *m2, int N);

}

#endif