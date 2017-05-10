// -*- C++ -*-

#include "stlib/mpi/Data.h"

#include <cassert>


using namespace stlib;

struct Pod
{
  double x;
  int y;
};


template<typename _T>
inline
void
testPod()
{
  mpi::Data<_T> const data;
  int size = 0;
  MPI_Type_size(data.type(), &size);
  assert(size == sizeof(_T));
}


template<typename _T>
inline
void
testBuiltin()
{
  int size;
  MPI_Type_size(mpi::Data<_T>::type(), &size);
  assert(size == sizeof(_T));
}


template<typename _T, std::size_t _N>
inline
void
testArray()
{
  mpi::Data<std::array<_T, _N> > data;
  MPI_Datatype const x = data.type();
  int size;
  MPI_Type_size(x, &size);
  assert(size == sizeof(std::array<_T, _N>));
}


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  {
    MPI_Datatype x = mpi::Data<char>::type();
    assert(x == MPI_CHAR);
  }

  {
    MPI_Datatype x = mpi::Data<signed char>::type();
    assert(x == MPI_SIGNED_CHAR);
  }
  {
    MPI_Datatype x = mpi::Data<unsigned char>::type();
    assert(x == MPI_UNSIGNED_CHAR);
  }

  {
    MPI_Datatype x = mpi::Data<short>::type();
    assert(x == MPI_SHORT);
  }
  {
    MPI_Datatype x = mpi::Data<unsigned short>::type();
    assert(x == MPI_UNSIGNED_SHORT);
  }

  {
    MPI_Datatype x = mpi::Data<int>::type();
    assert(x == MPI_INT);
  }
  {
    MPI_Datatype x = mpi::Data<unsigned int>::type();
    assert(x == MPI_UNSIGNED);
  }

  {
    MPI_Datatype x = mpi::Data<long>::type();
    assert(x == MPI_LONG);
  }
  {
    MPI_Datatype x = mpi::Data<unsigned long>::type();
    assert(x == MPI_UNSIGNED_LONG);
  }

  {
    MPI_Datatype x = mpi::Data<long long>::type();
    assert(x == MPI_LONG_LONG);
  }
  {
    MPI_Datatype x = mpi::Data<unsigned long long>::type();
    assert(x == MPI_UNSIGNED_LONG_LONG);
  }

  {
    MPI_Datatype x = mpi::Data<float>::type();
    assert(x == MPI_FLOAT);
  }
  {
    MPI_Datatype x = mpi::Data<double>::type();
    assert(x == MPI_DOUBLE);
  }

  //static_assert(std::is_pod<std::pair<double, int> >::value, "Error");
  testPod<Pod>();

  testBuiltin<float>();

  testArray<float, 3>();
  testArray<std::array<float, 3>, 3>();

  MPI_Finalize();
  return 0;
}
