#include <cuda_runtime.h>

#include "xpl-tracer.h"

int read(int data[][8], int x, int y)
{
  return data[x][y]; 
}

int write(int data[][8], int x, int y, int v)
{
  data[x][y] = v;
}

int& accessor(int data[][8], int x, int y)
{
  return data[x][y];
}

int valueAt(int data[][8], int x, int y)
{
  return accessor(data,x,y);
}

void setAt(int data[][8], int x, int y, int v)
{
  accessor(data,x,y) = v;
}

int main()
{
  int y[8][8];

	for (int i = 0; i < 4; ++i)
	  for (int j = 0; j < 4; ++j)
		{
	    write(y, i+0, j, i+j);
			setAt(y, i+4, j, i+j);
		}

  for (int i = 0; i < 4; ++i)
		for (int j = 4; j < 8; ++j)
		{
		  write(y, i+0, j, read(y, i+0, j-4));
			setAt(y, i+4, j, valueAt(y, i+4, j-4));					
		}
}
