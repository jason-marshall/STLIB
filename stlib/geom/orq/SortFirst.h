// -*- C++ -*-

/*!
  \file SortFirst.h
  \brief A class for a vector of records in N-D sorted in each dimension.
*/

#if !defined(__geom_SortFirst_h__)
#define __geom_SortFirst_h__

#include "stlib/geom/orq/ORQ.h"

#include <iostream>
#include <vector>
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
class SortFirst :
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
  //! A const iterator.
  typedef typename Container::const_iterator ConstIterator;

  //
  // Comparison functors.
  //
protected:

  //! Less than comparison in a specified coordinate.
  class LessThanCompare :
    public std::binary_function<typename Base::Record,
    typename Base::Record, bool>
  {
  private:
    typedef std::binary_function<typename Base::Record,
            typename Base::Record, bool> Bf;
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


  //! Less than comparison in the first coordinate for a record iterator and a multi-key.
  class LessThanCompareValueAndMultiKey :
    public std::binary_function<typename Base::Record, typename Base::Point,
    bool>
  {
  private:
    typedef std::binary_function<typename Base::Record, typename Base::Point,
            bool> Bf;
    _Location _f;

  public:

    //! Less than comparison in the first coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type value,
               const typename Bf::second_argument_type& multiKey) const
    {
      return _f(value)[0] < multiKey[0];
    }
  };


  //! Less than comparison in the first coordinate for a multi-key and a record iterator.
  class LessThanCompareMultiKeyAndValue :
    public std::binary_function<typename Base::Point, typename Base::Record,
    bool>
  {
  private:
    typedef std::binary_function<typename Base::Point, typename Base::Record,
            bool> Bf;
    _Location _f;

  public:

    //! Less than comparison in the first coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type& multiKey,
               const typename Bf::second_argument_type value) const
    {
      return  multiKey[0] < _f(value)[0];
    }
  };

  //
  // Member data.
  //
private:

  //! Records sorted by the first coordinate.
  Container _sorted;
  mutable LessThanCompare _lessThanCompare;
  mutable LessThanCompareValueAndMultiKey _lessThanCompareValueAndMultiKey;
  mutable LessThanCompareMultiKeyAndValue _lessThanCompareMultiKeyAndValue;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  SortFirst() :
    Base(),
    _sorted()
  {
  }

  //! Reserve storage for \c size records.
  explicit
  SortFirst(const std::size_t size) :
    Base(),
    _sorted()
  {
    _sorted.reserve(size);
  }

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  SortFirst(typename Base::Record first, typename Base::Record last) :
    Base(),
    _sorted()
  {
    insert(first, last);
    sort();
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Insert records.
  // @{

  //! Insert a single record.
  void
  insert(const typename Base::Record record)
  {
    _sorted.push_back(record);
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

  // @}
  //-------------------------------------------------------------------------
  //! \name Memory usage.
  // @{

  //! Return the total memory usage.
  std::size_t
  getMemoryUsage() const
  {
    return sizeof(SortFirst) + Base::size() * sizeof(typename Base::Record);
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Window Queries.
  // @{

  //! Sort the records by x coordinate.
  void
  sort()
  {
    std::sort(_sorted.begin(), _sorted.end(), _lessThanCompare);
  }

  //! Get the records in the window.  Return the # of records inside.
  template<typename OutputIterator>
  std::size_t
  computeWindowQuery(OutputIterator iter,
                     const typename Base::BBox& window) const;

  // @}
  //-------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the records.
  void
  put(std::ostream& out) const;

  // @}
  //-------------------------------------------------------------------------
  //! \name Validity check.
  // @{

  //! Check the validity of the data structure.
  bool
  isValid() const;

  //@}
};


//! Write to a file stream.
/*! \relates SortFirst */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const SortFirst<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_SortFirst_ipp__
#include "stlib/geom/orq/SortFirst.ipp"
#undef __geom_SortFirst_ipp__

#endif
