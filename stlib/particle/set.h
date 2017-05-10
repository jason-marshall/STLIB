// -*- C++ -*-

/*!
  \file particle/set.h
  \brief Use run-length encoding to store sets of particle indices.
*/

#if !defined(__particle_set_h__)
#define __particle_set_h__

#include "stlib/particle/types.h"

#include <algorithm>
#include <vector>

#include <cassert>

namespace stlib
{
namespace particle
{

//! A range of particles is represented with a particle index and an extent.
struct ParticleRange {
  //! The first particle in the range.
  /*! There should be no particle preceding this that has the same code.
    That is, it should be the first particle in some cell. */
  std::size_t first;
  //! The extent of the range.
  /*! The extent may span multiple cells. */
  std::size_t extent;
};

//! Compare by the first particle indices.
inline
bool
operator<(const ParticleRange& a, const ParticleRange& b)
{
  return a.first < b.first;
}


//! Use run-length encoding to store sets of particle indices.
class ParticleSet :
  public std::vector<ParticleRange>
{
  //
  // Types.
  //
public:

  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;

private:

  //! The base class is a vector of particle ranges.
  typedef std::vector<ParticleRange> Base;

  //
  // Member data.
  //
private:

  //! A buffer that is used for packing.
  Base _buffer;

  //
  // Member functions.
  //
public:

  //! Add a range of particles given the first particle in the cell.
  void
  append(std::size_t first, const std::vector<Code>& codes);

  //! Add a range of particles.
  void
  append(const std::size_t first, const std::size_t extent)
  {
    if (extent) {
      Base::push_back(ParticleRange{
        first, extent
      });
    }
  }

  //! Pack the set of particles to merge adjoining ranges.
  void
  pack();

  //! Offset all particle indices.
  void
  offset(const std::size_t n)
  {
    for (std::size_t i = 0; i != size(); ++i) {
      (*this)[i].first += n;
    }
  }

  //! Count the number of particles in the set.
  std::size_t
  cardinality() const
  {
    std::size_t n = 0;
    for (std::size_t i = 0; i != size(); ++i) {
      n += (*this)[i].extent;
    }
    return n;
  }

  //! Return true if the set is valid and packed.
  bool
  isValid() const;

  //! Copy the field values for the particles into a contiguous vector.
  template<typename _T, typename _Allocator>
  void
  copyField(const std::vector<_T>& field,
            std::vector<_T, _Allocator>* inSet) const;
};


} // namespace particle
}

#define __particle_set_tcc__
#include "stlib/particle/set.tcc"
#undef __particle_set_tcc__

#endif
