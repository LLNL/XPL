#include "xpl-tracer.h"

struct Base 
{
}
;

struct Derived : public Base
{
}
;

struct S 
{
  struct Derived _m_derived;
  Base &base();
}
;

Base &S::base()
{
  return ((this) -> _m_derived);
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
