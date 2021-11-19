#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct img 
{
  int len;
  int *red;
  int *green;
  int *blue;
}
;

struct img *createImg(int sz)
{
  return new ( img ) ({sz, (new int [sz]), (new int [sz]), (new int [sz])});
}

int main()
{
  struct img *image = createImg(16);
  struct img img2 = {(4), (new int [4]), (new int [4]), (new int [4])};
  
#pragma xpl diagnostic tracePrint(std::cout; image, img2)
  tracePrint(std::cout,XPLAllocData(image, "image"),XPLAllocData((image)->red, "(image)->red"),XPLAllocData((image)->green, "(image)->green"),XPLAllocData((image)->blue, "(image)->blue"),XPLAllocData((img2).red, "(img2).red"),XPLAllocData((img2).green, "(img2).green"),XPLAllocData((img2).blue, "(img2).blue"));
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
