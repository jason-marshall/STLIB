// -*- C++ -*-

/*!
  \file geom/spatialIndexing/OrthtreeElementNull.h
  \brief CONTINUE
*/

#if !defined(__geom_spatialIndexing_OrthtreeElementNull_h__)
#define __geom_spatialIndexing_OrthtreeElementNull_h__

namespace stlib
{
namespace geom
{

//! A null element.  It holds no data.
template<std::size_t _Dimension>
class
  OrthtreeElementNull
{
};


//! Return true.
/*!
  \relates OrthtreeElementNull
*/
template<std::size_t _Dimension>
inline
bool
operator==(const OrthtreeElementNull<_Dimension>& /*x*/,
           const OrthtreeElementNull<_Dimension>& /*y*/)
{
  return true;
}

//! Write nothing.
/*!
  \relates OrthtreeElementNull
*/
template<std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out, const OrthtreeElementNull<_Dimension>& /*x*/)
{
  return out;
}

//! Read nothing.
/*!
  \relates OrthtreeElementNull
*/
template<std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, OrthtreeElementNull<_Dimension>& /*x*/)
{
  return in;
}

//! Refinement functor that does nothing.
struct RefineNull {
  template<class _Orthtree>
  //! Do nothing.
  void
  operator()(const _Orthtree& /*orthtree*/,
             const typename _Orthtree::Element& /*parent*/,
             const std::size_t /*n*/,
             const typename _Orthtree::Key& /*key*/,
             typename _Orthtree::Element* /*child*/) const
  {
  }
};

//! Coarsening functor that does nothing.
struct CoarsenNull {
  template<class _Orthtree>
  //! Do nothing.
  void
  operator()
  (const _Orthtree& /*orthtree*/,
   const std::array<const typename _Orthtree::Element*,
   _Orthtree::NumberOfOrthants>& /*children*/,
   const typename _Orthtree::Key& /*key*/,
   typename _Orthtree::Element* /*parent*/) const
  {
  }
};

} // namespace geom
}

#endif
