// -*- C++ -*-

/*!
  \file PlaceboCheck.h
  \brief A placebo class.
*/

#if !defined(__geom_PlaceboCheck_h__)
#define __geom_PlaceboCheck_h__

#include "stlib/geom/orq/Placebo.h"

namespace stlib
{
namespace geom
{

//! A placebo for ORQ's in N-D
/*!
  Stores a vector of records.
*/
template<std::size_t N, typename _Location>
class PlaceboCheck :
  public Placebo<N, _Location>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef Placebo<N, _Location> Base;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors.
  //@{

  //! Default constructor.
  PlaceboCheck() :
    Base()
  {
  }

  //! Reserve storage for \c size records.
  explicit
  PlaceboCheck(const std::size_t size) :
    Base(size)
  {
  }

  //! Construct from a range of records.
  /*!
    \param first the beginning of a range of records.
    \param last the end of a range of records.
  */
  PlaceboCheck(typename Base::Record first, typename Base::Record last) :
    Base(first, last)
  {
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
};

//! Write to a file stream.
/*! \relates PlaceboCheck */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const PlaceboCheck<N, _Location>& x)
{
  x.put(out);
  return out;
}

} // namespace geom
}

#define __geom_PlaceboCheck_ipp__
#include "stlib/geom/orq/PlaceboCheck.ipp"
#undef __geom_PlaceboCheck_ipp__

#endif
