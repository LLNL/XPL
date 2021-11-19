#include <cuda_runtime.h>
#include "xpl-tracer.h"

void setPtrVal(int *x,int val)
{
  traceW( *x) = val;
}

int main()
{
  int xyz[100];
  
#pragma xpl diagnostic tracerPrint(std::cerr)
  tracerPrint(std::cerr);
  setPtrVal(xyz + 1,0);
  
#pragma xpl diagnostic tracerPrint(std::cout)
  tracerPrint(std::cout);
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
