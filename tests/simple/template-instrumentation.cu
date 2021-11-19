#include <cuda_runtime.h>

#include "xpl-tracer.h"

template <class T>
static inline
T*
gpuAlloc(size_t num)
{
  void* ptr = NULL;

	cudaMallocManaged(&ptr, sizeof(T) * num);
	return reinterpret_cast<T*>(ptr);
}


int main()
{
  int* x = gpuAlloc<int>(8);

	return 0;
}
