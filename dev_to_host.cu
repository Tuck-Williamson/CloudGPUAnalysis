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

//template <typename TK>
#define TK unsigned int
__global__ void k(TK *d, size_t n){

  if(threadIdx.x == 0 && blockIdx.x == 0){
    printf("\tTest message: %s\n", devMessage);
    printf("\tMessage Location: %p\n", devMessage);
    printf("\tGrid Dim: %d\n", gridDim.x);
    printf("\tBlock Dim: %d\n", blockDim.x);
  }
  else{
    //printf(" %d.%d", threadIdx.x, blockIdx.x);
  }
  __brkpt();
  TK *myCode = (TK*)&devMessage[0];
  for (size_t i = blockIdx.x*blockDim.x+threadIdx.x; i < n; i+=gridDim.x*blockDim.x)
    d[i] = myCode[i];
}

template <typename T>
void printMem(T *d, size_t n){
  //std::cout << std::hex;
  for(uint i = 0; i < n; i++){
    //std::cout << "(" << std::dec << i << "): " << std::hex << d[i]  << " ";
    printf("(%6d): %08x ", i, d[i]);
    if( (i+1) % 8 == 0 ){
        //std::cout << std::endl;
        printf("\n");
    }
  }
}

char gc(char curChar){
  if(31 < curChar && curChar < 127)
    return curChar;
  else
    return '.';
}

template <typename T>
void printMemChar(T *d, size_t n, uint numEleNLine){
  //std::cout << std::hex;
  for(uint i = 0; i < n; i+=numEleNLine){
    char* data = (char*)&d[i];
    unsigned int dIdx = 0;
    printf("(%6d): ", i);
    for ( uint curIdx = 0; curIdx < numEleNLine; curIdx++){
        printf("%08x ", d[i+curIdx]);
    }
    printf("|");
    for ( uint curIdx = 0; curIdx < numEleNLine; curIdx++){
        printf("%c%c%c%c", gc(data[dIdx]), gc(data[dIdx+1]), gc(data[dIdx+2]), gc(data[dIdx+3]));
        dIdx += 4;
    }

    printf("\n");
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
  /*
  */

  //Allocate host pinned data buffer dest.
  unsigned int *d;
  size_t n = 1*1024;
  cuEChk(cudaHostAlloc(&d, sizeof(d[0])*n, cudaHostAllocDefault));

  //k<<<160, 1024>>>(d, n);
  k<<<1, 1024>>>(d, n);
  cuEChk(cudaDeviceSynchronize());
  std::cout << "Print before dev to host memcopy.\n";
  //printMemChar(d, n, 8);
  std::cout << "\n\n";
  //unsigned int *d1;
  //cudaMalloc(&d1, sizeof(d[0])*n);
  //cudaMemcpy(d, d1, sizeof(d[0])*n, cudaMemcpyDeviceToHost);
  //printMem(d1, n);
}