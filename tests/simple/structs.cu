#include <cstring>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "xpl-tracer.h"

struct X 
{ 
  int     posMin;
	int     posMax;
	int     sz;
	int     capacity;
	double  valMin;
	double  valMax;
	double* values;	 
};


X create()
{
  X x = { 0, 0, 0, 0, 0.0, 0.0, 0 };  

  return x;
}

double* allocmem(int sz)
{
  double* mem = 0;

  cudaMallocManaged((void**)&mem, sizeof(double) * sz, 0);
	return mem;
}

void resize(X* x)
{
  if (x->capacity == 0)
	{
	  x->capacity = 4;
		x->values = allocmem(x->capacity);
		return;
	}

	int     capacity_ = x->capacity*2;
	double* values_ = allocmem(capacity_);

	memcpy(values_, x->values, sizeof(x->values[0]) * x->capacity);
  x->values   = values_;
	x->capacity = capacity_;
}

void ensure_capacity(X* x)
{
  if (x->sz == x->capacity)
	  resize(x);
}

void add(X* x, double v)
{
  ensure_capacity(x);

	x->values[x->sz++] = v;

	if (x->sz == 1)
	{
	  x->valMin = x->valMax = v;
		x->posMin = x->posMax = 0;
	}
  else if (x->valMin > v)
	{
	  x->valMin = v;
		x->posMin = x->sz-1;
	}
  else if (x->valMax < v)
	{
	  x->valMax = v;
		x->posMax = x->sz-1;
	}
}

X copy(X* x)
{
  X o = *x;

	o.values = allocmem(o.capacity);
  memcpy(o.values, x->values, sizeof(x->values[0]) * o.capacity);
  return o;
}

int* szPtr(X& x)
{
  int z = x.sz;
  x.sz = z/(z-1);
	return &x.sz;
}

int main()
{
  X   x;

	add(&x, 3.3);
	add(&x, 4.4);

  X   y = copy(&x);
  
	add(&y, 5.5);

	int* psz = szPtr(x);
  return 0;
}
