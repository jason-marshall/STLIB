// -*- C++ -*-

#include <cassert>

struct A {
  const int x;

  A() :
    x()
  {
  }

  A(const int y) :
    x(y)
  {
  }
};

int
main()
{
  A a(23);
  // The default copy constructor is fine.
  A b = a;
  assert(a.x == b.x);
#if 0
  // This won't work because the default assigment operator won't be defined.
  A c;
  c = a;
  assert(a.x == c.x);
#endif

  return 0;
}
