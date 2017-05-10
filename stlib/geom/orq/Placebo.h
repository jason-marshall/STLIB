// -*- C++ -*-

/*!
  \file Placebo.h
  \brief A placebo class.
*/

#if !defined(__geom_Placebo_h__)
#define __geom_Placebo_h__

#include "stlib/geom/orq/ORQ.h"

#include "stlib/numerical/random/uniform/DiscreteUniformGeneratorMt19937.h"

#include <vector>
#include <algorithm>

namespace stlib
{
namespace geom
{


//! A placebo for ORQ's in N-D
/*!
  Stores a vector of records.
*/
template<std::size_t N, typename _Location>
class Placebo :
  public Orq<N, _Location>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef Orq<N, _Location> Base;

  //
  // Member data.
  //
protected:

  //! The vector of pointers to records.
  std::vector<typename Base::Record> _records;
  //! Random number generator.
  mutable numerical::DiscreteUniformGeneratorMt19937 _random;

public:
  //! The number of records to return with each query.
  std::size_t querySize;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  //@{

  //! Default constructor.
  Placebo() :
    Base(),
    _records(),
    querySize(0) {}

  //! Reserve storage for \c size records.
  explicit
  Placebo(const std::size_t size) :
    Base(),
    _records(),
    querySize(0)
  {
    _records.reserve(size);
  }

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  Placebo(typename Base::Record first, typename Base::Record last) :
    Base(),
    _records(),
    querySize(0)
  {
    insert(first, last);
    shuffle();
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Shuffle the record pointers.
  void
  shuffle()
  {
    std::random_shuffle(_records.begin(), _records.end());
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Insert records.
  //@{

  //! Insert a single record.
  void
  insert(typename Base::Record record)
  {
    _records.push_back(record);
    ++Base::_size;
  }

  //! Insert a range of records.
  void
  insert(typename Base::Record first, typename Base::Record last)
  {
    while (first != last) {
      insert(first);
      ++first;
    }
    shuffle();
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Window queries.
  //@{

  //! Get the records in the window.  Return the # of records inside.
  template<typename _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const typename Base::BBox& window) const;

  //@}
  //-------------------------------------------------------------------------
  //! \name Memory usage.
  //@{

  //! Return the total memory usage.
  std::size_t
  getMemoryUsage() const
  {
    return (sizeof(std::vector<typename Base::Record>) +
            _records.size() * sizeof(typename Base::Record));
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Print the records.
  void
  put(std::ostream& out) const;

  //@}

protected:

  //! Return a starting point for the window query.
  std::size_t
  getStartingPoint() const
  {
    return _random() % (_records.size() - querySize);
  }
};


//! Write to a file stream.
/*! \relates Placebo */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const Placebo<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_Placebo_ipp__
#include "stlib/geom/orq/Placebo.ipp"
#undef __geom_Placebo_ipp__

#endif
