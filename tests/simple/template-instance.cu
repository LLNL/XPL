#include "xpl-tracer.h"

struct S
{
    template<typename T>
		void x(T& t)
		{
		  deref(&t);
		}

	  template<typename T>
		T deref(T* t)
		{
		  return *t; 
		}
};

int main()
{
  int n = 0;
  S   s;

	s.x(n);
}


