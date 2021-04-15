
#include "../header/kernel.h"
#include <eigen3/Eigen/Core>

#include <iostream>
#include <stdio.h>


static void HandleError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// CUDA Version
namespace Kernel
{
    __global__ void cu_dot(Eigen::Vector3d *v1, Eigen::Vector3d *v2, double *out, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
        {
            out[idx] = v1[idx].dot(v2[idx]);
        }
        return;
    }

    // The wrapper for the calling of the actual kernel
    double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2)
    {        
        int n = v1.size();
        double *ret = new double[n];
        // Allocate device arrays
        Eigen::Vector3d *dev_v1, *dev_v2;
        HANDLE_ERROR(cudaMalloc((void **)&dev_v1, sizeof(Eigen::Vector3d)*n));
        HANDLE_ERROR(cudaMalloc((void **)&dev_v2, sizeof(Eigen::Vector3d)*n));
        double* dev_ret;
        HANDLE_ERROR(cudaMalloc((void **)&dev_ret, sizeof(double)*n));

        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_v1, v1.data(), sizeof(Eigen::Vector3d)*n, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_v2, v2.data(), sizeof(Eigen::Vector3d)*n, cudaMemcpyHostToDevice));

        // Dot product
        cu_dot<<<(n+1023)/1024, 1024>>>(dev_v1, dev_v2, dev_ret, n);
        
        // Copy to host
        HANDLE_ERROR(cudaMemcpy(ret, dev_ret, sizeof(double)*n, cudaMemcpyDeviceToHost));

        // Reduction of the array
        for (int i=1; i<n; ++i)
        {
            ret[0] += ret[i];
        }

        // Return
        return ret[0];
    }

    /*-------------------- dot product matrix ---------------------*/
    __global__ void cu_dotMatrix(const Eigen::ArrayXXd *m1, const Eigen::ArrayXXd *m2, Eigen::ArrayXXd *out, size_t N)
    {
        int ROW = blockIdx.y*blockDim.y+threadIdx.y;
        int COL = blockIdx.x*blockDim.x+threadIdx.x;      

        double tmpSum = 0;
        
        if (ROW < N && COL < N) {
            // each thread computes one element of the block sub-matrix
            for (int i = 0; i < N; i++) {
                tmpSum += (*m1)(ROW * N + i) * (*m2)(i * N + COL);
            }
        }
        (*out)(ROW * N + COL) = tmpSum; 
        return;
    }
    
    // The wrapper for the calling of the actual kernel
    Eigen::MatrixXd dotMatrix(const Eigen::ArrayXXd & m1, const Eigen::ArrayXXd  & m2)
    {        
        int n1 = m1.size();
        int n2 = m2.size();

        //Instantiate
        Eigen::ArrayXXd *dev_m1, *dev_m2;
        //Alloc on GPU 
        HANDLE_ERROR(cudaMalloc((void **)&dev_m1, sizeof(double)*n1));
        HANDLE_ERROR(cudaMalloc((void **)&dev_m2, sizeof(double)*n2));
        
        //Instantiate
        Eigen::ArrayXXd* dev_ret;
        //Alloc on GPU
        HANDLE_ERROR(cudaMalloc((void **)&dev_ret, sizeof(double)*n2));

        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_m1, m1.data(), sizeof(double)*n1, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_m2, m2.data(), sizeof(double)*n2, cudaMemcpyHostToDevice));
    
        // Dot product
        cu_dotMatrix<<<(n1+1023)/1024, 1024>>>(dev_m1, dev_m2, dev_ret, n1);
                
        // Copy to host
        //HANDLE_ERROR(cudaMemcpy(ret, dev_ret, sizeof(double)*n, cudaMemcpyDeviceToHost));


        return m1;
    }
}

