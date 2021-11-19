#include "xpl-tracer.h"

const int* addr(const int& i)
{
  return &i;
}

int main()
{
  int  i = 0;
	int* p = &i;

	const int* cp = addr(*p);
	return 0;
}

