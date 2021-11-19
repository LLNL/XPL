#include <cuda_runtime.h>
#include "xpl-tracer.h"

int read(int data[][8],int x,int y)
{
  return traceR(data[x][y]);
}

int write(int data[][8],int x,int y,int v)
{
  traceW(data[x][y]) = v;
}

int &accessor(int data[][8],int x,int y)
{
  return data[x][y];
}

int valueAt(int data[][8],int x,int y)
{
  return traceR((accessor(data,x,y)));
}

void setAt(int data[][8],int x,int y,int v)
{
  traceW((accessor(data,x,y))) = v;
}

int main()
{
  int y[8][8];
  for (int i = 0; i < 4; traceR(++i)) 
    for (int j = 0; j < 4; traceR(++j)) {
      write(y,i + 0,j,i + j);
      setAt(y,i + 4,j,i + j);
    }
  for (int i = 0; i < 4; traceR(++i)) 
    for (int j = 4; j < 8; traceR(++j)) {
      write(y,i + 0,j,(read(y,i + 0,j - 4)));
      setAt(y,i + 4,j,(valueAt(y,i + 4,j - 4)));
    }
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
