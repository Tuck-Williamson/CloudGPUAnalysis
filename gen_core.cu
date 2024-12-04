#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#define cuEChk(toCheck) errChk(toCheck, __LINE__)
void errChk(cudaError_t status, size_t line){
    if(status != cudaSuccess){
        std::cerr << "There was a cuda error at line " << line << "." << std::endl;
        std::cerr << "Error (" << status << "): " << cudaGetErrorName(status) << "::" << cudaGetErrorString(status) << std::endl;
        throw 1;
    }
}

struct DataStruct{
    const char data[1024*1023];
};

__device__ char* devMessage= "\t\tHello from GPU.\n";

__global__ void findOffsets(long *startOff, long *endOff){

    DataStruct *data = (DataStruct*)alloca(sizeof(DataStruct));
    printf("\tMessage Location: %p\n", devMessage);
    printf("\tStack Location: %p\n", &startOff);

    char testChar;
    char* curPtr = devMessage;
    uint jumpBy = 16;
    while(1==1){
        startOff[0]+=jumpBy;
        testChar = curPtr[-startOff[0]];
    }

}

int main(){

  long *startOff = 0;
  long *endOff = 0;
  cuEChk(cudaHostAlloc(&startOff, sizeof(startOff[0])*2, cudaHostAllocDefault));
  endOff = &startOff[1];
  startOff[0] = 0;
  endOff[0] = 0;

  findOffsets<<<1,1>>>(startOff, endOff);
  cudaDeviceSynchronize();
  printf("Start Offset: %ld.\n", startOff[0]);
  printf("End Offset: %ld.\n", endOff[0]);
  cuEChk(cudaDeviceSynchronize());

}