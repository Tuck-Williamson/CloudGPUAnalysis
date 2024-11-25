//From https://developer.nvidia.com/blog/even-easier-introduction-cuda/
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#define cuEChk(toCheck) errChk(toCheck, __LINE__)
void errChk(cudaError_t status, size_t line){
    if(status != cudaSuccess){
        std::cerr << "There was a cuda error at line " << line << "." << std::endl;
        std::cerr << "Error (" << status << "): " << cudaGetErrorName(status) << "::" << cudaGetErrorString(status) << std::endl;
        throw 1;
    }
}

__device__ char* devMessage= "\t\tHello from GPU.\n";

// Kernel function to add the elements of two arrays
__global__ void dev_tester(int dummy)
{
    if(threadIdx.x == 0){
        printf("Hello from thread: %d.\n", threadIdx.x);
        printf("\tTest message: %s\n", devMessage);
        printf("\tMessage Location: %p\n", devMessage);
        printf("\tKernel Location: %p\n", dev_tester);
        printf("\tExecution Stack Location: %p\n", &dummy);
    }


}

int main(int argc, char *argv[])
{
  dev_tester<<<1, 1>>>(42);

  // Wait for GPU to finish before accessing on host
  cuEChk(cudaDeviceSynchronize());

  return 0;
}