#include <cstdlib>
#include <cuda_runtime.h>
#include "xpl-tracer.h"
template < class Fn >
void exec ( Fn fn, int ofs )
{
  fn ( ofs );
}

void add_cst(double *mat,int x,int y,int c)
{
  for (int i = 0; i < x; traceR(++i)) 
    for (int j = 0; j < y; traceR(++j)) {
      int ofs = j * x + i;
      traceW(mat[ofs]) = ((double )i);
      traceRW( *(mat + ofs)) += ((double )j);
      exec( [mat,c] (int ofs) -> void 
{
        traceRW(traceR((this) -> mat)[ofs]) += ((double )(traceR((this) -> c)));
      },ofs);
    }
}

int main()
{
  static const int sz = 16;
  double *matrix = (double *)(calloc((sz * sz),sizeof(double )));
  add_cst(matrix,sz,sz,3);
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
