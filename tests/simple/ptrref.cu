#include <cuda_runtime.h>

#include "xpl-tracer.h"

void setPtrVal(int* x, int val)
{
  *x = val;
}

int readPtrVal(int* x)
{
  int y = *x;
}

int& returnRef(int* x)
{
  return *x;
}

int returnVal(int* x)
{
  return *x;
}

int** returnPtrPtr(int*& x)
{
  return &x;
}

int main()
{
  int xyz[100];

  setPtrVal(xyz+1, 0);
  readPtrVal(xyz);
  returnRef(xyz);
  returnRef(xyz) = 1;
  int y = returnRef(xyz);

  y = returnRef(xyz) + 1;
  int z = returnVal(xyz);
  z = returnVal(xyz);

  int*  yp  = &y;
	int** ypp = returnPtrPtr(yp);

  return 0;
}
