#include <cuda_runtime.h>

#include "xpl-tracer.h"

struct img
{
  int   len;
  int*  red;
	int*  green;
	int*  blue;
};

img* createImg(int sz)
{
  return new img{ sz, new int[sz], new int[sz], new int[sz] };
}

int main()
{
  img* image = createImg(16);
  img  img2{ 4, new int[4], new int[4], new int[4] };

  #pragma xpl diagnostic tracePrint(std::cout; image, img2)
  return 0;
}
