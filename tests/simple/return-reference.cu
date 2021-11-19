#include "xpl-tracer.h"

struct Base {};

struct Derived : Base {};

struct S
{
  Derived _m_derived;

	Base& base(); 
};

Base&
S::base() { return _m_derived; }
