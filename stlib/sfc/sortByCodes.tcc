// -*- C++ -*-

#if !defined(__sfc_sortByCodes_tcc__)
#error This file is an implementation detail of sortByCodes.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Grid, typename _Object>
inline
void
sortByCodes(_Grid const& order, std::vector<_Object>* objects,
            std::vector<std::pair<typename _Grid::Code,
            std::size_t> >* codeIndexPairs)
{
  typedef typename _Grid::BBox BBox;

  // Allocate memory for the code/index pairs.
  codeIndexPairs->resize(objects->size());
  // Calculate the codes.
  for (std::size_t i = 0; i != codeIndexPairs->size(); ++i) {
    (*codeIndexPairs)[i].first =
      order.code(centroid(geom::specificBBox<BBox>((*objects)[i])));
    (*codeIndexPairs)[i].second = i;
  }
  // Sort by the codes.
  lorg::sort(codeIndexPairs, order.numBits());
  // Set the sorted objects.
  {
    std::vector<_Object> obj(codeIndexPairs->size());
    for (std::size_t i = 0; i != codeIndexPairs->size(); ++i) {
      obj[i] = (*objects)[(*codeIndexPairs)[i].second];
    }
    objects->swap(obj);
  }
}


template<typename _Grid, typename _Object>
inline
void
sortByCodes(_Grid const& order, std::vector<_Object>* objects,
            std::vector<typename _Grid::Code>* codes)
{
  std::vector<std::pair<typename _Grid::Code, std::size_t> >
    codeIndexPairs;
  sortByCodes(order, objects, &codeIndexPairs);
  codes->resize(codeIndexPairs.size());
  for (std::size_t i = 0; i != codes->size(); ++i) {
    (*codes)[i] = codeIndexPairs[i].first;
  }
}


template<typename _Grid, typename _Object>
inline
void
sortByCodes(_Grid const& order, std::vector<_Object>* objects)
{
  // Make a temporary vector of code/index pairs.
  std::vector<std::pair<typename _Grid::Code, std::size_t> >
    codeIndexPairs;
  sortByCodes(order, objects, &codeIndexPairs);
}


template<typename _Code, typename _Object>
inline
void
sortByCodes(std::vector<_Object>* objects, OrderedObjects* orderedObjects)
{
  // The default BBox type for the object. We use this to deduce the dimension
  // and floating-point number type.
  typedef typename geom::BBoxForObject<void, _Object>::DefaultBBox BBox;
  // The traits for the space-filling curve.
  typedef Traits<BBox::Dimension, typename BBox::Float, _Code> Traits;
  // The grid for the SFC.
  typedef LocationCode<Traits> Grid;

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
  
  // Record the original order of the objects if necessary.
  if (orderedObjects != nullptr) {
    orderedObjects->set(codeIndexPairs);
  }
}


} // namespace sfc
} // namespace stlib
