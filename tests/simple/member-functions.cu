#include <utility>
#include <cuda_runtime_api.h>
#include "xpl-tracer.h"


double* allocmem(int sz)
{
  double* mem = 0;

  cudaMallocManaged((void**)&mem, sizeof(double) * sz, 0);
	return mem;
}

void freemem(double* mem)
{
  if (mem == 0) return;

  cudaFree(mem);
}


struct DoubleArray 
{ 
  DoubleArray()
	: posMin(0), posMax(0), sz(0), capacity(0), valMin(0), valMax(0), values(nullptr)
	{}

	DoubleArray(const DoubleArray& orig)
	: posMin(orig.posMin), posMax(orig.posMax),
	  sz(orig.sz), capacity(orig.capacity), 
		valMin(orig.valMin), valMax(orig.valMax),
		values(allocmem(capacity))
	{
	  cudaMemcpy(this->values, orig.values, sizeof(values[0]) * sz, cudaMemcpyDeviceToDevice);
	}

  ~DoubleArray()
	{
	  freemem(values);
	}

	double& at(int pos)
	{
	  return values[pos];
	}

	void add(double v);
	void resize();
	void ensure_capacity();

  int     posMin;
	int     posMax;
	int     sz;
	int     capacity;
	double  valMin;
	double  valMax;
	double* values;	 
};


void DoubleArray::resize()
{
	int     capacity_ = capacity ? capacity*2 : 4;
	double* values_   = allocmem(capacity_);

	cudaMemcpy(values_, values, sizeof(values[0]) * capacity,  cudaMemcpyDeviceToDevice);
  
	std::swap(this->values, values_);
	this->capacity = capacity_;
	freemem(values_);
}

void DoubleArray::ensure_capacity()
{
  if (sz == capacity)
	  resize();
}

void DoubleArray::add(double v)
{
  ensure_capacity();

	values[sz++] = v;

	if (sz == 1)
	{
	  valMin = valMax = v;
		posMin = posMax = 0;
	}
  else if (valMin > v)
	{
	  valMin = v;
		posMin = sz-1;
	}
  else if (valMax < v)
	{
	  valMax = v;
		posMax = sz-1;
	}
}

int main()
{
  DoubleArray x;

	x.add(3.3);
	x.add(4.4);

  DoubleArray y(x);
  
	y.add(5.5);
  return 0;
}
