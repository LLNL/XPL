#include <cuda_runtime.h>

#include "xpl-tracer.h"

struct X { int a; float b; };

int minElemVal(X* x)
{
  if (x == 0) return 0;

  if (x->a > (*x).b)
    return x->b;

  return (*x).a;
}


int main()
{
  X   data{};
  X*  x = &data;
  int i = minElemVal(x);

  return 0;
}
