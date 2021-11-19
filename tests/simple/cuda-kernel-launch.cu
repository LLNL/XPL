#include <cuda_runtime.h>

#include "xpl-tracer.h"


__global__
void clearImageSequence(int* root)
{
  const int x = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	*(root+x) = 0;
}


__managed__ int img[64*64*10];

int main()
{
  clearImageSequence<<<1,1>>>(img);
  
  return 0;
}
