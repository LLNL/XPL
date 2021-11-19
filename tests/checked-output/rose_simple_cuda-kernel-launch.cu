#include <cuda_runtime.h>
#include "xpl-tracer.h"

__global__ void clearImageSequence(int *root)
{
  const int x = (int )(blockIdx . x * blockDim . x * blockDim . y + threadIdx . y * blockDim . x + threadIdx . x);
  traceW( *(root + x)) = 0;
}
int img[40960];

int main()
{
  clearImageSequence<<<1,1>>>(img);
  cudaDeviceSynchronize();
  return 0;
}

template<> inline __gnu_cxx::__enable_if< true,void > ::__type std::__fill_a(_Bit_type *__first,_Bit_type *__last,const int &__value)
{
  const int __tmp = traceR(__value);
  for (; __first != __last; traceR(++__first)) 
    traceW( *__first) = ((_Bit_type )__tmp);
}

template<> inline void std::fill(_Bit_type *__first,_Bit_type *__last,const int &__value)
{
  ;
  __fill_a((__niter_base(__first)),(__niter_base(__last)),(traceR(__value)));
}
