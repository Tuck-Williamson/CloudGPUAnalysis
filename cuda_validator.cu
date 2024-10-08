//From https://developer.nvidia.com/blog/even-easier-introduction-cuda/
#include <cuda.h>
#include <iostream>
#include <iomanip>

// Kernel function to add the elements of two arrays
__global__ void validation_gen(const unsigned int maxIdx, unsigned int* dataPtr)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int idx = index; idx < maxIdx; idx += stride)
      dataPtr[idx] = stride << 16 & index;
}

int main(int argc, char *argv[])
{
  unsigned int N = 256<<2;
  unsigned int* dataPtr;

  // Allocate Unified Memory - accessible from CPU or GPU
  cudaMallocManaged(&dataPtr, N*sizeof(dataPtr[0]));

  // Run kernel on 1M elements on the GPU
  validation_gen<<<1, 256>>>(N, dataPtr);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  std::cout << std::hex;
  for(uint i = 0; i < N; i++){
    auto idx = i & 0xffff80;
    auto stride = i & 0x7f;
    stride = stride << 16-5;

    std::cout << "(" << std::dec << i << "): " << std::hex << dataPtr[i] << " -> " << stride << idx;
    if(i % 4 == 0){
        std::cout << std::endl;
    }
  }

  // Free memory
  cudaFree(dataPtr);

  return 0;
}