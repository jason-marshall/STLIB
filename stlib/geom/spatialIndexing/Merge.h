// -*- C++ -*-

/*!
  \file geom/spatialIndexing/Merge.h
  \brief Merging functors.
*/

#if !defined(__geom_spatialIndexing_Merge_h__)
#define __geom_spatialIndexing_Merge_h__

namespace stlib
{
namespace geom
{

//! Merging functor that does nothing.
struct MergeNull {
  //! Do nothing.
  void
  operator()() const
  {
  }
};

//! Merging functor that copies the value of the first child.
struct MergeCopyFirst {
  //! Copy the value of the first child.
  template<std::size_t NumberOfOrthants, typename _Element>
  void
  operator()
  (const std::array<const _Element*, NumberOfOrthants>& children,
   _Element* parent) const
  {
    *parent = *children[0];
  }
};

//! Merging functor that averages the values of the children.
struct MergeAverage {
  //! Average the children.
  template<std::size_t NumberOfOrthants, typename _Element>
  void
  operator()
  (const std::array<const _Element*, NumberOfOrthants>& children,
   _Element* parent) const
  {
    *parent = *children[0];
    for (std::size_t i = 0; i != NumberOfOrthants; ++i) {
      *parent == *children[i];
    }
    *parent /= NumberOfOrthants;
  }
};

} // namespace geom
}

#endif
