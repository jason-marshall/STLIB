// -*- C++ -*-

#if !defined(__particle_set_tcc__)
#error This file is an implementation detail of set.
#endif

namespace stlib
{
namespace particle
{


inline
void
ParticleSet::
append(const std::size_t first, const std::vector<Code>& codes)
{
  // The code for the neighbor particle.
  const Code code = codes[first];
  // Count the number of particles in this cell.
  std::size_t extent = 1;
  for (; codes[first + extent] == code; ++extent) {
  }
  append(first, extent);
}


inline
void
ParticleSet::
pack()
{
  // Check the trivial case.
  if (empty()) {
    return;
  }
  // Sort by the first particle indices.
  std::sort(begin(), end());
  // Pack into the buffer and then swap.
  _buffer.clear();
  _buffer.push_back(front());
  for (std::size_t i = 1; i != size(); ++i) {
    if (_buffer.back().first + _buffer.back().extent ==
        (*this)[i].first) {
      _buffer.back().extent += (*this)[i].extent;
    }
    else {
      _buffer.push_back((*this)[i]);
    }
  }
  swap(_buffer);
}


inline
bool
ParticleSet::
isValid() const
{
  if (empty()) {
    return true;
  }
  for (std::size_t i = 0; i != size(); ++i) {
    if ((*this)[i].extent == 0) {
      return false;
    }
  }
  for (std::size_t i = 0; i != size() - 1; ++i) {
    if ((*this)[i].first >= (*this)[i + 1].first) {
      return false;
    }
  }
  return true;
}


} // namespace particle
}
