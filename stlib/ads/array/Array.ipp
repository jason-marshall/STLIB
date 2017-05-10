// -*- C++ -*-

#if !defined(__Array_ipp__)
#error This file is an implementation detail of the class Array.
#endif

namespace stlib
{
namespace ads
{


//
// Manipulators
//


template<int N, typename T, bool A>
inline
void
Array<N, T, A>::
resize(const index_type& ext)
{
  // If the number of elements differs.
  const int numberOfElements = computeProduct(ext);
  if (numberOfElements != size()) {
    container_base::resize(numberOfElements);
  }
  // If the extents are not the same.
  if (ext != extents()) {
    indexing_base::rebuild(ext, data());
  }
}


template<int N, typename T, bool A>
inline
void
Array<N, T, A>::
resize(const range_type& rng)
{
  // If the number of elements differs.
  if (rng.content() != size()) {
    container_base::resize(rng.content());
  }
  // if the ranges are not the same.
  if (rng != ranges()) {
    indexing_base::rebuild(rng, data());
  }
}


//
// I/O member functions.
//


template<int N, typename T, bool A>
inline
void
Array<N, T, A>::
get(std::istream& in, std::true_type)
{
  range_type rng;
  in >> rng;
  resize(rng);
  read_elements_ascii(in);
}


template<int N, typename T, bool A>
inline
void
Array<N, T, A>::
get(std::istream& in, std::false_type)
{
  range_type rng;
  in >> rng;
  assert(rng == ranges());
  read_elements_ascii(in);
}

template<int N, typename T, bool A>
inline
void
Array<N, T, A>::
read(std::istream& in, std::true_type)
{
  range_type rng;
  in.read(reinterpret_cast<char*>(&rng), sizeof(range_type));
#ifdef STLIB_DEBUG
  assert(size_type(in.gcount()) == sizeof(range_type));
#endif
  resize(rng);
  read_elements_binary(in);
}


template<int N, typename T, bool A>
inline
void
Array<N, T, A>::
read(std::istream& in, std::false_type)
{
  range_type rng;
  in.read(reinterpret_cast<char*>(&rng), sizeof(range_type));
#ifdef STLIB_DEBUG
  assert(size_type(in.gcount()) == sizeof(range_type));
#endif
  assert(rng == ranges());
  read_elements_binary(in);
}

} // namespace ads
} // namespace stlib
