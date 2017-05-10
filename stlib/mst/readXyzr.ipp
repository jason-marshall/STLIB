// -*- C++ -*-

#if !defined(__mst_readXyzr_ipp__)
#error This file is an implementation detail of readXyzr.
#endif

namespace stlib
{
namespace mst
{

template < typename T, typename PointOutputIterator,
           typename NumberOutputIterator >
inline
void
readXyzr(const char* fileName, PointOutputIterator points,
         NumberOutputIterator radii)
{
  std::ifstream inputStream(fileName);
  readXyzr<T>(inputStream, points, radii);
}


template<typename T>
inline
bool
getXyzr(std::istream& inputStream, std::array<T, 3>* point, T* radius)
{
  inputStream >> *point >> *radius;
  return inputStream.good();
}


template < typename T, typename PointOutputIterator,
           typename NumberOutputIterator >
inline
void
readXyzr(std::istream& inputStream, PointOutputIterator points,
         NumberOutputIterator radii)
{
  typedef T Number;
  typedef std::array<T, 3> Point;

  Point point;
  Number radius;
  while (getXyzr(inputStream, &point, &radius)) {
    *points = point;
    ++points;
    *radii = radius;
    ++radii;
  }
}





template<typename T, typename AtomOutputIterator>
inline
void
readXyzr(const char* fileName, AtomOutputIterator atoms)
{
  std::ifstream inputStream(fileName);
  readXyzr<T>(inputStream, atoms);
}


template<typename T>
inline
bool
getXyzr(std::istream& inputStream, geom::Ball<T, 3>* atom)
{
  inputStream >> *atom;
  return inputStream.good();
}


template<typename T, typename AtomOutputIterator>
inline
void
readXyzr(std::istream& inputStream, AtomOutputIterator atoms)
{
  typedef T Number;
  typedef geom::Ball<Number, 3> AtomType;

  AtomType atom;
  while (getXyzr<Number>(inputStream, &atom)) {
    *atoms = atom;
    ++atoms;
  }
}


} // namespace mst
}
