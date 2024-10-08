#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>

void errChk(cudaError_t status, size_t line){
    if(status != cudaSuccess){
        std::cerr << "There was a cuda error at line " << line << "." << std::endl;
        std::cerr << "Error (" << status << "): " << cudaGetErrorName(status) << "::" << cudaGetErrorString(status) << std::endl;
        throw 1;
    }
}

#define MByte 1024*1024

// __global__ void validationGen(const unsigned int* endPtr, unsigned int* basePtr)
// {
//   auto index = threadIdx.x;
//   auto stride = blockDim.x;
//   for (int i = index; i < n; i += stride)
//       y[i] = x[i] + y[i];
// }

int main(int numArgs, char* args[]) {

    cudaError_t cuErr;

    cudaDeviceProp props;
    errChk(cudaGetDeviceProperties(&props, 0), __LINE__);

    size_t totalMem = props.totalGlobalMem;

    void* devPtr;
    std::cout << "Trying to allocate " << totalMem/MByte << " M bytes." << std::endl;
    while(cudaMalloc(&devPtr, totalMem) != cudaSuccess)
    {
        cuErr = cudaPeekAtLastError();
        if(totalMem < 5*MByte || cuErr != cudaErrorMemoryAllocation){
            std::cerr << "There was a cuda error when allocating " << totalMem << " bytes." << std::endl;
            errChk(cuErr, __LINE__);
            return 2;
        }
        totalMem -= MByte;
        std::cout << "Trying to allocate " << totalMem/MByte << " M bytes." << std::endl;
    }
    std::cout << "Successfully allocated " << totalMem << " bytes." << std::endl;


    char* hostPtr = (char*)malloc(totalMem);
    try{
        errChk(cudaMemcpy((void*)hostPtr, devPtr, totalMem, cudaMemcpyDeviceToHost), __LINE__);

        std::cout << "Read " << totalMem << " bytes from GPU memory." << std::endl;

        errChk(cudaFree(devPtr), __LINE__);

        std::cout << "Writing " << totalMem << " bytes to gdump.bin file." << std::endl;
        FILE* dumpFilePtr = fopen("gdump.bin", "wb");
        if(dumpFilePtr == NULL){
            std::cerr << "Error opening 'gdump.bin'!" << std::endl;
            return 1;
        }

        // Optimizations for sparse file generation.

        //Check larger sections using 64b
        unsigned long long* startDataPtr = reinterpret_cast<unsigned long long*>(hostPtr);
        unsigned long long* endHostPtr = reinterpret_cast<unsigned long long*>(hostPtr+totalMem);
        unsigned long long* endDataPtr = startDataPtr;

        while(endDataPtr < endHostPtr){
          //Get ptr to first non-zero data.
          startDataPtr = std::find_if(startDataPtr,
                                endHostPtr, [](auto datum){ return datum != 0;});
          //Get ptr to first zero data after non-zero data.
          endDataPtr = std::find_if(static_cast<unsigned long long*>(startDataPtr),
                                endHostPtr, [](auto datum){ return datum == 0;});

          std::cout << "  Writing " << (endDataPtr - startDataPtr) * sizeof(*endDataPtr)
                << " bytes offset by " << (char*)startDataPtr - hostPtr << " from " << startDataPtr << " to " << endDataPtr << "." << std::endl;
          fseek(dumpFilePtr, (char*)startDataPtr - hostPtr, SEEK_SET);
          fwrite(startDataPtr, sizeof(*startDataPtr), endDataPtr-startDataPtr, dumpFilePtr);
          fflush(dumpFilePtr);
        }
        //fwrite(hostPtr, 1, totalMem, dumpFilePtr);

        fclose(dumpFilePtr);


        free(hostPtr);
    }
    catch(int err){
        cudaFree(devPtr);
        free(hostPtr);
        return 1;
    }
    return 0;
}