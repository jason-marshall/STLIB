// -*- C++ -*-

/*!
  \file particle/lookup.h
  \brief Lookup table for a sorted array of codes.
*/

#if !defined(__particle_lookup_h__)
#define __particle_lookup_h__

#include "stlib/particle/types.h"

#include <functional>
#include <limits>
#include <vector>

#include <cassert>

namespace stlib
{
namespace particle
{


//! Lookup table for a sorted array of codes.
class LookupTable :
  public std::unary_function<IntegerTypes::Code, std::size_t>
{

  //
  // Types.
  //
public:

  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;
private:

  typedef std::unary_function<Code, std::size_t> Base;

public:

  //! The argument type is a code.
  typedef Base::argument_type argument_type;
  //! The result type is an index into the array of codes.
  typedef Base::result_type result_type;

  //
  // Member data.
  //
private:

  //! The table of indices.
  std::vector<std::size_t> _table;
  //! The right shift to apply to codes before indexing.
  int _shift;
  //! The first (shifted) code in the table.
  Code _first;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Default constructor results in an empty table.
  /*! You must call initialize() before using this functor. */
  LookupTable()
  {
    clear();
  }

  //! Construct from the sorted array of codes and a maximum table size.
  /*! This constructor just calls initialize(). */
  LookupTable(const std::vector<Code>& codes, std::size_t maxSize)
  {
    initialize(codes, maxSize);
  }

  //! Initialize from the sorted array of codes and a maximum table size.
  /*! Note that the vector of codes has a guard element. */
  void
  initialize(const std::vector<Code>& codes, std::size_t maxSize);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! The right shift that is applied to codes before indexing.
  /*! If the shift is zero, then the table uses exact lookup. Otherwise,
   it gives you a lower bound on the index. */
  int
  shift() const
  {
    return _shift;
  }

  //! Return the memory usage in bytes.
  std::size_t
  memoryUsage() const
  {
    return _table.size() * sizeof(std::size_t);
  }

  //! Return the memory capacity in bytes.
  std::size_t
  memoryCapacity() const
  {
    return _table.capacity() * sizeof(std::size_t);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Return the index of the first code that is not less than the argument.
  result_type
  operator()(argument_type code) const;

  //! Clear the data structure.
  void
  clear();

  //@}
};


} // namespace particle
}

#define __particle_lookup_tcc__
#include "stlib/particle/lookup.tcc"
#undef __particle_lookup_tcc__

#endif
