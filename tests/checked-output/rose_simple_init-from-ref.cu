#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct X 
{
  int posMin;
  int posMax;
}
;

struct X create(const int &a,const int &b)
{
  int min = traceR(a);
  struct X x = {min, traceR(b)};
  return x;
}

int main()
{
  int z = 0;
  int &y = z;
  struct X x = create((traceR(y)),z);
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
