#include <cuda_runtime.h>

#include "xpl-tracer.h"

struct int_duple
{
  int_duple&
  operator=(const int_duple& other)
  {
    lhs = other.lhs;
    rhs = other.rhs;
    return *this;
  }

  int lhs;
  int rhs;
};

