// -*- C++ -*-

/*!
  \file SortProject.h
  \brief A class for a vector of records in N-D sorted in each dimension.
*/

#if !defined(__geom_SortProject_h__)
#define __geom_SortProject_h__

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
class SortProject :
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
    public std::binary_function<typename Base::Record, typename Base::Record,
    bool>
  {
  private:
    typedef std::binary_function<typename Base::Record, typename Base::Record,
            bool> Bf;
    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor.  The coordinate to compare has an invalid value.
    LessThanCompare() :
      _n(-1) {}

    //! Set the coordinate to compare.
    void
    set(const std::size_t n)
    {
      _n = n;
    }

    //! Less than comparison in a specified coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type x,
               const typename Bf::second_argument_type y) const
    {
      return _f(x)[_n] < _f(y)[_n];
    }
  };


  //! Less than comparison in a specified coordinate for a record iterator and a multi-key.
  class LessThanCompareValueAndMultiKey :
    public std::binary_function<typename Base::Record, typename Base::Point,
    bool>
  {
  private:
    typedef std::binary_function<typename Base::Record, typename Base::Point,
            bool> Bf;
    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor.  The coordinate to compare has an invalid value.
    LessThanCompareValueAndMultiKey() :
      _n(-1) {}

    //! Set the coordinate to compare.
    void
    set(const std::size_t n)
    {
      _n = n;
    }

    //! Less than comparison in a specified coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type value,
               const typename Bf::second_argument_type multiKey) const
    {
      return _f(value)[_n] < multiKey[_n];
    }
  };


  //! Less than comparison in a specified coordinate for a multi-key and a record iterator.
  class LessThanCompareMultiKeyAndValue :
    public std::binary_function<typename Base::Point, typename Base::Record,
    bool>
  {
  private:
    typedef std::binary_function<typename Base::Point, typename Base::Record,
            bool> Bf;
    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor.  The coordinate to compare has an invalid value.
    LessThanCompareMultiKeyAndValue() :
      _n(-1) {}

    //! Set the coordinate to compare.
    void
    set(const std::size_t n)
    {
      _n = n;
    }

    //! Less than comparison in a specified coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type multiKey,
               const typename Bf::second_argument_type value) const
    {
      return  multiKey[_n] < _f(value)[_n];
    }
  };

  //
  // Member data.
  //

private:

  //! Pointers to elements sorted by each coordinate.
  std::array<Container, N> _sorted;
  mutable LessThanCompare _lessThanCompare;
  mutable LessThanCompareValueAndMultiKey _lessThanCompareValueAndMultiKey;
  mutable LessThanCompareMultiKeyAndValue _lessThanCompareMultiKeyAndValue;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  SortProject() :
    Base(),
    _sorted()
  {
  }

  //! Reserve storage for \c size records.
  explicit
  SortProject(const std::size_t size) :
    Base(),
    _sorted()
  {
    for (std::size_t n = 0; n != N; ++n) {
      _sorted[n].reserve(size);
    }
  }

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  SortProject(typename Base::Record first, typename Base::Record last) :
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
    for (std::size_t n = 0; n != N; ++n) {
      _sorted[n].push_back(record);
    }
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
    return sizeof(SortProject) + N * Base::size() *
           sizeof(typename Base::Record);
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Window Queries.
  // @{

  //! Sort the records by x, y and z coordinate.
  void
  sort()
  {
    // Sort in each direction.
    for (std::size_t n = 0; n != N; ++n) {
      _lessThanCompare.set(n);
      std::sort(_sorted[n].begin(), _sorted[n].end(), _lessThanCompare);
    }
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
/*! \relates SortProject */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const SortProject<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_SortProject_ipp__
#include "stlib/geom/orq/SortProject.ipp"
#undef __geom_SortProject_ipp__

#endif
