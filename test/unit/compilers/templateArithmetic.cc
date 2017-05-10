// -*- C++ -*-

template<int N, int M>
class A {};

// The IBM compiler does not correctly implement some template arithmetic.
// It cannot understand the meaning of M - 1.  Below is a hack you can apply
// to get the program to compile.

#ifdef __IBM_ATTRIBUTES
template<int N, int M, int M_1>
void
f(A<N, M> a, A<N, M_1> b)
{
  static_assert(M - 1 == M_1, "Bad template parameter values.");
}
#else
template<int N, int M>
void
f(A<N, M> /*a*/, A < N, M - 1 > /*b*/) {}
#endif

int
main()
{
  A<2, 2> a;
  A<2, 1> b;
  f(a, b);

  return 0;
}
