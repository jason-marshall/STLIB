// -*- C++ -*-

/*!
  \file SortRankProject.h
  \brief A class for a coordinate sorted and ranked vector of grid elements
  in N-D.  The class implements the point-in-box method.
*/

#if !defined(__geom_SortRankProject_h__)
#define __geom_SortRankProject_h__

#include "stlib/geom/orq/ORQ.h"

#include <vector>
#include <algorithm>

namespace stlib
{
namespace geom
{

//! A sorted and ranked vector of grid elements in N-D.
/*!
  Coordinate sorted and ranked vector of records.

  This class implements the point-in-box method developed by Swegle,
  et. al.
*/
template<std::size_t N, typename _Location>
class SortRankProject :
  public Orq<N, _Location>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef Orq<N, _Location> Base;

  typedef typename std::vector<typename Base::Record>::const_pointer
  const_pointer;

  //
  // Comparison functors.
  //
private:

  //! Less than comparison in a specified coordinate for pointers to records.
  class LessThanCompare :
    public std::binary_function<const_pointer, const_pointer, bool>
  {
  private:
    typedef std::binary_function<const_pointer, const_pointer, bool> Base;
    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor. The coordinate to compare has an invalid value.
    LessThanCompare() :
      _n(-1) {}

    //! Set the coordinate to compare.
    void
    set(const std::size_t n)
    {
      _n = n;
    }

    //! Less than comparison in a specified coordinate.
    typename Base::result_type
    operator()(const typename Base::first_argument_type x,
               const typename Base::second_argument_type y) const
    {
      return _f(*x)[_n] < _f(*y)[_n];
    }
  };


  //! Less than comparison in a specified coordinate for a pointer to a record and a multi-key.
  class LessThanComparePointerAndMultiKey :
    public std::binary_function<const_pointer, typename Base::Point, bool>
  {
  private:
    typedef std::binary_function<const_pointer, typename Base::Point, bool>
    Bf;
    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor.  The coordinate to compare has an invalid value.
    LessThanComparePointerAndMultiKey() :
      _n(-1) {}

    //! Set the coordinate to compare.
    void
    set(const std::size_t n)
    {
      _n = n;
    }

    //! Less than comparison in a specified coordinate.
    typename Bf::result_type
    operator()(const typename Bf::first_argument_type recordPointer,
               const typename Bf::second_argument_type multiKey) const
    {
      return _f(*recordPointer)[_n] < multiKey[_n];
    }
  };


  //! Less than comparison in a specified coordinate for a multi-key and a pointer to record iterator.
  class LessThanCompareMultiKeyAndPointer :
    public std::binary_function<typename Base::Point, const_pointer, bool>
  {
  private:
    typedef std::binary_function<typename Base::Point, const_pointer, bool>
    Bf;
    std::size_t _n;
    _Location _f;

  public:

    //! Default constructor.  The coordinate to compare has an invalid value.
    LessThanCompareMultiKeyAndPointer() :
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
               const typename Bf::second_argument_type recordPointer)
    const
    {
      return  multiKey[_n] < _f(*recordPointer)[_n];
    }
  };

  //
  // Member data.
  //
private:

  //! The vector of pointers to records.
  std::vector<typename Base::Record> _records;
  //! Pointers to elements sorted by each coordinate.
  std::array<std::vector<const_pointer>, N> _sorted;
  //! The rank of the records in each coordinate.
  std::array<std::vector<std::size_t>, N> _rank;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  SortRankProject();

  //! Reserve storage for \c size records.
  explicit
  SortRankProject(const std::size_t size);

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  SortRankProject(typename Base::Record first, typename Base::Record last);

  // @}
  //-------------------------------------------------------------------------
  //! \name Insert records.
  // @{

  //! Insert a single record.
  void
  insert(const typename Base::Record record)
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
  }

  // @}
  //-------------------------------------------------------------------------
  //! \name Memory usage.
  // @{

  //! Return the total memory usage.
  std::size_t
  getMemoryUsage() const;

  // @}
  //-------------------------------------------------------------------------
  //! \name Window Queries.
  // @{

  //! Sort and rank the records in each coordinate.
  void
  sortAndRank();

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
  //-------------------------------------------------------------------------
  //! \name Validity check.
  // @{

  //! Check the validity of the data structure.
  bool
  isValid() const;

  //@}
};

//! Write to a file stream.
/*! \relates SortRankProject */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const SortRankProject<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_SortRankProject_ipp__
#include "stlib/geom/orq/SortRankProject.ipp"
#undef __geom_SortRankProject_ipp__

#endif
