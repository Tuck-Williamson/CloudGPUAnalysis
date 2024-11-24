//From https://developer.nvidia.com/blog/even-easier-introduction-cuda/
#include <cuda.h>
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

// Kernel function to add the elements of two arrays
__global__ void validation_gen(const unsigned int maxIdx, unsigned int* dataPtr)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int idx = index; idx < maxIdx; idx += stride)
      dataPtr[idx] = stride << 16 | index;
}

int main(int argc, char *argv[])
{
  unsigned int N = 256<<2;
  unsigned int* dataPtr;
  const size_t memSize = N*sizeof(dataPtr[0]);

  // Allocate Unified Memory - accessible from CPU or GPU
//   cudaMallocManaged(&dataPtr, N*sizeof(dataPtr[0]));
  cuEChk(cudaMalloc(&dataPtr, memSize));

  // Run kernel on 1M elements on the GPU
  validation_gen<<<1, 256>>>(N, dataPtr);

  // Wait for GPU to finish before accessing on host
  cuEChk(cudaDeviceSynchronize());

  unsigned short* hostPtr = (unsigned short*)malloc(memSize);
  cuEChk(cudaMemcpy((void*)hostPtr, dataPtr, memSize, cudaMemcpyDeviceToHost));

  std::cout << std::hex;
  for(uint i = 0; i < N; i++){
    auto idx = i % 256;
    auto stride = i / 256;

    std::cout << "(" << std::dec << i << "): " << std::hex << hostPtr[i*2] << hostPtr[i*2+1] << " -> " << stride << idx << " ";
    if(i % 4 == 0){
        std::cout << std::endl;
    }
  }

  std::cout << std::endl << std::endl;

  // Free memory
  cuEChk(cudaFree(dataPtr));

  return 0;
}