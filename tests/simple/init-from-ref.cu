#include <cstring>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct X 
{ 
  int     posMin;
	int     posMax;
};


X create(const int& a, const int& b)
{
  int min = a;
  X   x = { min, b };

  return x;
}

int main()
{
  int  z = 0;
  int& y = z;
  X    x = create(y, z);
}
