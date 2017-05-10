// -*- C++ -*-

#include <numeric>
#include <iostream>
#include <cstdio>

int
main()
{
  //char* a = static_cast<char*>(::operator new(100000000 * sizeof(char)));
  const std::size_t size = 100000000;
  char* a = new char[size];
  std::fill(a, a + size, ' ');
  getchar();
  //std::cerr << std::accumulate(a, a + size, 0.) << '\n';
  delete[] a;
  //char* a = static_cast<char*>(::operator new(10000000 * sizeof(char)));
  //::operator delete(a);
  getchar();

  return 0;
}
