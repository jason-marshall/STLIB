// -*- C++ -*-

/*!
  \file SortFirstDynamic.h
  \brief A class for a vector of records in N-D sorted in each dimension.
*/

#if !defined(__geom_SortFirstDynamic_h__)
#define __geom_SortFirstDynamic_h__

#include "stlib/geom/orq/ORQ.h"

#include "stlib/ads/algorithm/sort.h"

#include <iostream>
#include <map>
#include <algorithm>

namespace stlib
{
namespace geom
{

//! A sorted vector of records in N-D.
/*!
  Dimension sorted vectors of records.
*/
template<std::size_t N, typename _Location>
class SortFirstDynamic :
  public Orq<N, _Location>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef Orq<N, _Location> Base;
  //! The container of record iterators.
  typedef std::multimap<typename Base::Float, typename Base::Record> Container;
  //! A const iterator.
  typedef typename Container::value_type Value;
  typedef typename Container::iterator Iterator;
  typedef typename Container::const_iterator ConstIterator;

protected:

  //
  // Comparison functors.
  //

  //! Less than comparison in a specified coordinate.
  class LessThanCompare :
    public std::binary_function<typename Base::Record, typename Base::Record,
    bool>
  {
  private:
    typedef std::binary_function<typename Base::Record, typename Base::Record,
            bool> Bf;
    _Location _f;

  public:

    //! Less than comparison in the first coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type x,
               const typename Bf::second_argument_type y) const
    {
      return _f(x)[0] < _f(y)[0];
    }
  };

  //
  // Member data.
  //

private:

  //! Records sorted by the first coordinate.
  Container _records;
  mutable LessThanCompare _lessThanCompare;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  SortFirstDynamic() :
    Base(),
    _records()
  {
  }

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  SortFirstDynamic(typename Base::Record first, typename Base::Record last) :
    Base(),
    _records()
  {
    insert(first, last);
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Insert records.
  // @{

  //! Insert a single record.
  void
  insert(const typename Base::Record record)
  {
    _records.insert(Value(Base::_location(record)[0], record));
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
  }

  //! Erase a record.
  /*!
    The record must exist.
  */
  void
  erase(const typename Base::Record record)
  {
    const typename Base::Float key = Base::_location(record)[0];
    Iterator i = _records.lower_bound(key);
    assert(i != _records.end());
    // For each record with this key.
    for (; i->first == key; ++i) {
      if (i->second == record) {
        _records.erase(i);
        --Base::_size;
        return;
      }
    }
    assert(false);
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
/*! \relates SortFirstDynamic */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const SortFirstDynamic<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_SortFirstDynamic_ipp__
#include "stlib/geom/orq/SortFirstDynamic.ipp"
#undef __geom_SortFirstDynamic_ipp__

#endif
