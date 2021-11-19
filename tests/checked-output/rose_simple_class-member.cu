#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct int_duple 
{
  

  inline int_duple &operator=(const struct int_duple &other)
{
    traceW((this) -> lhs) = traceR(other . lhs);
    traceW((this) -> rhs) = traceR(other . rhs);
    return  *(this);
  }
  int lhs;
  int rhs;
}
;

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
