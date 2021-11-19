#include <cstdlib>
#include <cuda_runtime.h>

#include "xpl-tracer.h"

template <class Fn>
void exec(Fn fn, int ofs)
{
  fn(ofs);
}

void add_cst(double* mat, int x, int y, int c)
{
  for (int i = 0; i < x; ++i)
    for (int j = 0; j < y; ++j)
    {
		  int ofs = j*x+i;

			mat[ofs] = i;
			*(mat+ofs) += j; 
      exec( [mat, c](int ofs) -> void
            {
		 			    mat[ofs] += c;
					  },
					  ofs
				  );
		}
}

int main()
{
  static const int sz = 16;

  double* matrix = (double*) calloc(sz*sz, sizeof(double));

  add_cst(matrix, sz, sz, 3);
  return 0;
}
