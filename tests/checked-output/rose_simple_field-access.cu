#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct X 
{
  int a;
  float b;
}
;

int minElemVal(struct X *x)
{
  if (x == ((struct X *)0)) 
    return 0;
  if (((float )(traceR(x -> a))) > traceR(traceR( *x) . b)) 
    return (int )(traceR(x -> b));
  return traceR(traceR( *x) . a);
}

int main()
{
  struct X data = {};
  struct X *x = &data;
  int i = minElemVal(x);
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
