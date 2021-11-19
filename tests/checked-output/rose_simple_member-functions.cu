#include <utility>
#include <cuda_runtime_api.h>
#include "xpl-tracer.h"

double *allocmem(int sz)
{
  double *mem = (double *)0;
  traceMallocManaged((void **)(&mem),sizeof(double ) * ((unsigned long )sz),(unsigned int )0);
  return mem;
}

void freemem(double *mem)
{
  if (mem == ((double *)0)) 
    return ;
  traceFree((void *)mem);
}

struct DoubleArray 
{
  

  inline DoubleArray() : posMin(0), posMax(0), sz(0), capacity(0), valMin(0), valMax(0), values((nullptr))
{
  }
  

  inline DoubleArray(const struct DoubleArray &orig) : posMin(traceR(orig . posMin)), posMax(traceR(orig . posMax)), sz(traceR(orig . sz)), capacity(traceR(orig . capacity)), valMin(traceR(orig . valMin)), valMax(traceR(orig . valMax)), values(allocmem((traceR((this) -> capacity))))
{
    traceMemcpy((void *)(traceR((this) -> values)),(const void *)(traceR(orig . values)),sizeof((this) -> values[0]) * ((unsigned long )(traceR((this) -> sz))),cudaMemcpyDeviceToDevice);
  }
  

  inline ~DoubleArray()
{
    freemem((traceR((this) -> values)));
  }
  

  inline double &at(int pos)
{
    return traceR((this) -> values)[pos];
  }
  void add(double v);
  void resize();
  void ensure_capacity();
  int posMin;
  int posMax;
  int sz;
  int capacity;
  double valMin;
  double valMax;
  double *values;
}
;

void DoubleArray::resize()
{
  int capacity_ = ((bool )(traceR((this) -> capacity)))?traceR((this) -> capacity) * 2 : 4;
  double *values_ = allocmem(capacity_);
  traceMemcpy((void *)values_,(const void *)(traceR((this) -> values)),sizeof((this) -> values[0]) * ((unsigned long )(traceR((this) -> capacity))),cudaMemcpyDeviceToDevice);
  swap((traceR((this) -> values)),values_);
  traceW((this) -> capacity) = capacity_;
  freemem(values_);
}

void DoubleArray::ensure_capacity()
{
  if (traceR((this) -> sz) == traceR((this) -> capacity)) 
    (this) ->  resize ();
}

void DoubleArray::add(double v)
{
  (this) ->  ensure_capacity ();
  traceW(traceR((this) -> values)[traceRW((this) -> sz)++]) = v;
  if (traceR((this) -> sz) == 1) {
    traceW((this) -> valMin) = traceR(traceW((this) -> valMax) = v);
    traceW((this) -> posMin) = traceR(traceW((this) -> posMax) = 0);
  }
   else if (traceR((this) -> valMin) > v) {
    traceW((this) -> valMin) = v;
    traceW((this) -> posMin) = traceR((this) -> sz) - 1;
  }
   else if (traceR((this) -> valMax) < v) {
    traceW((this) -> valMax) = v;
    traceW((this) -> posMax) = traceR((this) -> sz) - 1;
  }
}

int main()
{
  struct DoubleArray x;
  x .  add (3.3);
  x .  add (4.4);
  struct DoubleArray y(x);
  y .  add (5.5);
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

template<> inline std::enable_if< true,void > ::type std::swap(double *&__a,double *&__b)
{
  double *__tmp = move(__a);
  traceW(__a) = move(__b);
  traceW(__b) = move(__tmp);
}
