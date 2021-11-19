#include <cuda_runtime.h>

#include "xpl-tracer.h"

void setPtrVal(int* x, int val)
{
  *x = val;
}

int main()
{
  int xyz[100];

  #pragma xpl diagnostic tracerPrint(std::cerr)
  setPtrVal(xyz+1, 0);

  #pragma xpl diagnostic tracerPrint(std::cout)
  return 0;
}
