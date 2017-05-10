// -*- C++ -*-

#include <vector>

#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

void
f(std::vector<std::vector<std::size_t> >& a)
{
  std::size_t ThreadNumber = 0;
#ifdef _OPENMP
  ThreadNumber = omp_get_thread_num();
#endif
  std::vector<std::size_t>& x = a[ThreadNumber];
  x.resize(10000 * (ThreadNumber + 1));
  for (std::size_t i = 0; i != x.size(); ++i) {
    x[i] = ThreadNumber;
  }
  #pragma omp barrier
  for (std::size_t i = 0; i != a.size(); ++i) {
    assert(a[i][a[i].size() - 1] == i);
  }
}

int
main()
{
  std::size_t NumberOfThreads = 1;
#ifdef _OPENMP
  omp_set_num_threads(2);
  #pragma omp parallel
  if (omp_get_thread_num() == 0) {
    NumberOfThreads = omp_get_num_threads();
  }
#endif

  std::vector<std::vector<std::size_t> > a(NumberOfThreads);

  #pragma omp parallel
  {
    std::size_t ThreadNumber = 0;
#ifdef _OPENMP
    ThreadNumber = omp_get_thread_num();
#endif
    std::vector<std::size_t>& x = a[ThreadNumber];
    x.resize(10000 * (ThreadNumber + 1));
    for (std::size_t i = 0; i != x.size(); ++i) {
      x[i] = ThreadNumber;
    }
    #pragma omp barrier
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(a[i][a[i].size() - 1] == i);
    }
  }

  for (std::size_t i = 0; i != a.size(); ++i) {
    std::fill(a[i].begin(), a[i].end(), 0);
  }

  #pragma omp parallel
  f(a);

  return 0;
}
