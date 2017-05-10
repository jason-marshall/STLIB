// -*- C++ -*-

#if !defined(__Array1_ipp__)
#error This file is an implementation detail of the class Array.
#endif

namespace stlib
{
namespace ads
{


//
// Manipulators
//


template<typename T, bool A>
inline
void
Array<1, T, A>::
resize(const size_type sz)
{
  if (sz != size()) {
    container_base::resize(sz);
    indexing_base::rebuild(sz, data());
  }
}


template<typename T, bool A>
inline
void
Array<1, T, A>::
resize(const range_type& rng)
{
  if (size() != rng.extent()) {
    container_base::resize(rng.extent());
    indexing_base::rebuild(rng, data());
  }
  else if (range() != rng) {
    indexing_base::rebuild(rng, data());
  }
}


//
// I/O
//


template<typename T, bool A>
inline
void
Array<1, T, A>::
get(std::istream& in, std::true_type)
{
  range_type rng;
  in >> rng;
  resize(rng);
  read_elements_ascii(in);
}


template<typename T, bool A>
inline
void
Array<1, T, A>::
get(std::istream& in, std::false_type)
{
  range_type rng;
  in >> rng;
  assert(rng.extent() == size());
  read_elements_ascii(in);
}


template<typename T, bool A>
inline
void
Array<1, T, A>::
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


template<typename T, bool A>
inline
void
Array<1, T, A>::
read(std::istream& in, std::false_type)
{
  range_type rng;
  in.read(reinterpret_cast<char*>(&rng), sizeof(range_type));
#ifdef STLIB_DEBUG
  assert(size_type(in.gcount()) == sizeof(range_type));
#endif
  assert(rng.extent() == size());
  // Read the elements.
  read_elements_binary(in);
}


template<typename T, bool A>
inline
std::istream&
operator>>(std::istream& in, Array<1, T, A>& x)
{
  x.get(in);
  return in;
}

} // namespace ads
} // namespace stlib
