#include <cuda_runtime.h>

#include "xpl-tracer.h"

#pragma xpl diagnostic-ignore initialPrint, tracePrint, finalPrint

void setPtrVal(int* x, int val)
{
  #pragma xpl diagnostic initialPrint(std::cerr)
  #pragma xpl diagnostic tracerPrint(std::cerr)
  
  *x = val;
  #pragma xpl diagnostic finalPrint(std::cout)
}

#pragma xpl diagnostic-ignore 

int main()
{
  #pragma xpl diagnostic initialPrint(std::cerr)
  int xyz[100];

  #pragma xpl diagnostic tracerPrint(std::cerr)
  setPtrVal(xyz+1, 0);

  #pragma xpl diagnostic finalPrint(std::cout)
  return 0;
}
