// -*- C++ -*-

#if !defined(__sfc_sortByHilbertCodes_tcc__)
#error This file is an implementation detail of sortByHilbertCodes.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Code, typename _Object>
void
sortByHilbertCodes(std::vector<_Object>* objects)
{
  // The default BBox type for the object. We use this to deduce the dimension
  // and floating-point number type.
  using BBox = typename geom::BBoxForObject<void, _Object>::DefaultBBox;
  // The traits for the space-filling curve.
  using Traits =
    Traits<BBox::Dimension, typename BBox::Float, _Code, HilbertOrder>;
  // The grid for the SFC.
  using Grid = LocationCode<Traits>;

  // First check the trivial case.
  if (objects->empty()) {
    return;
  }

  // Put a bounding box around the objects. Then use the bounding box
  // to define the SFC grid.
  Grid const grid(geom::bbox(objects->begin(), objects->end()));

  // Generate codes for the objects and then sort by the codes.
  std::vector<std::pair<_Code, std::size_t> > codeIndexPairs;
  sortByCodes(grid, objects, &codeIndexPairs);
}


} // namespace sfc
} // namespace stlib
