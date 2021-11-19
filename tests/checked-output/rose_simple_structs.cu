#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct X 
{
  int posMin;
  int posMax;
  int sz;
  int capacity;
  double valMin;
  double valMax;
  double *values;
}
;

struct X create()
{
  struct X x = {(0), (0), (0), (0), (0.0), (0.0), (0)};
  return x;
}

double *allocmem(int sz)
{
  double *mem = (double *)0;
  traceMallocManaged((void **)(&mem),sizeof(double ) * ((unsigned long )sz),(unsigned int )0);
  return mem;
}

void resize(struct X *x)
{
  if (traceR(x -> capacity) == 0) {
    traceW(x -> capacity) = 4;
    traceW(x -> values) = allocmem((traceR(x -> capacity)));
    return ;
  }
  int capacity_ = traceR(x -> capacity) * 2;
  double *values_ = allocmem(capacity_);
  traceCMemcpy((void *)values_,(const void *)(traceR(x -> values)),sizeof(x -> values[0]) * ((unsigned long )(traceR(x -> capacity))));
  traceW(x -> values) = values_;
  traceW(x -> capacity) = capacity_;
}

void ensure_capacity(struct X *x)
{
  if (traceR(x -> sz) == traceR(x -> capacity)) 
    resize(x);
}

void add(struct X *x,double v)
{
  ensure_capacity(x);
  traceW(traceR(x -> values)[traceRW(x -> sz)++]) = v;
  if (traceR(x -> sz) == 1) {
    traceW(x -> valMin) = traceR(traceW(x -> valMax) = v);
    traceW(x -> posMin) = traceR(traceW(x -> posMax) = 0);
  }
   else if (traceR(x -> valMin) > v) {
    traceW(x -> valMin) = v;
    traceW(x -> posMin) = traceR(x -> sz) - 1;
  }
   else if (traceR(x -> valMax) < v) {
    traceW(x -> valMax) = v;
    traceW(x -> posMax) = traceR(x -> sz) - 1;
  }
}

struct X copy(struct X *x)
{
  struct X o = traceR( *x);
  traceW(o . values) = allocmem((traceR(o . capacity)));
  traceCMemcpy((void *)(traceR(o . values)),(const void *)(traceR(x -> values)),sizeof(x -> values[0]) * ((unsigned long )(traceR(o . capacity))));
  return o;
}

int *szPtr(struct X &x)
{
  int z = traceR(x . sz);
  traceW(x . sz) = z / (z - 1);
  return &x . sz;
}

int main()
{
  struct X x;
  add(&x,3.3);
  add(&x,4.4);
  struct X y = copy(&x);
  add(&y,5.5);
  int *psz = szPtr(x);
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
