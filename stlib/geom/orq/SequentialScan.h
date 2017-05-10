// -*- C++ -*-

/*!
  \file SequentialScan.h
  \brief The sequential scan algorithm.
*/

#if !defined(__geom_SequentialScan_h__)
#define __geom_SequentialScan_h__

#include "stlib/geom/orq/ORQ.h"

#include <vector>
#include <algorithm>

namespace stlib
{
namespace geom
{

//! The sequential scan algorithm for ORQ's in N-D.
/*!
  Stores a vector records.
*/
template<std::size_t N, typename _Location>
class SequentialScan :
  public Orq<N, _Location>
{
  //
  // Types.
  //

private:

  //! The base class.
  typedef Orq<N, _Location> Base;
  //! The container of records.
  typedef std::vector<typename Base::Record> Container;
  //! A const iterator over records.
  typedef typename Container::const_iterator ConstIterator;

  //
  // Member data.
  //

private:

  //! The vector of records.
  Container _records;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  SequentialScan() :
    Base(),
    _records() {}

  //! Reserve storage for \c size records.
  explicit
  SequentialScan(const std::size_t size) :
    Base(),
    _records()
  {
    _records.reserve(size);
  }

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  SequentialScan(typename Base::Record first, typename Base::Record last) :
    Base(),
    _records()
  {
    insert(first, last);
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Insert records.
  // @{

  //! Add a single record.
  void
  insert(const typename Base::Record record)
  {
    _records.push_back(record);
    ++Base::_size;
  }

  //! Add a range of records.
  void
  insert(typename Base::Record first, typename Base::Record last)
  {
    while (first != last) {
      insert(first);
      ++first;
    }
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Memory usage.
  // @{

  //! Return the total memory usage.
  std::size_t
  getMemoryUsage() const
  {
    return (sizeof(SequentialScan) +
            _records.size() * sizeof(typename Base::Record));
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Window Queries.
  // @{

  //! Get the records in the window.  Return the # of records inside.
  template<typename _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const typename Base::BBox& window) const;

  // @}
  //-------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the records.
  void
  put(std::ostream& out) const;

  // @}
};


//! Write to a file stream.
/*! \relates SequentialScan */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const SequentialScan<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_SequentialScan_ipp__
#include "stlib/geom/orq/SequentialScan.ipp"
#undef __geom_SequentialScan_ipp__

#endif
