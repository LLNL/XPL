#include <cuda.h>
#include <cuda_runtime.h>
#include "xpl-tracer.h"

int main()
{
  int* x;

	cudaMallocManaged(&x, sizeof(x));
}
