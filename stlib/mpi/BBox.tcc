// -*- C++ -*-

#if !defined(__mpi_BBox_tcc__)
#error This file is an implementation detail of BBox.
#endif

namespace stlib
{
namespace mpi
{


template<typename _Float, std::size_t _Dimension>
inline
geom::BBox<_Float, _Dimension>
reduce(geom::BBox<_Float, _Dimension> const& input, MPI_Comm comm, int root)
{
  typedef geom::BBox<_Float, _Dimension> BBox;

  BBox output = geom::specificBBox<BBox>();
  // Just gather all of the bounding boxes to the root and then merge them.
  // This is probably faster than a binary reduction.
  std::vector<BBox> const gathered = gather(input, comm, root);
  if (commRank(comm) == 0) {
    // Bound the bounding boxes. (Don't use the bound member function as it 
    // does correctly interpret empty bounding boxes.)
    output = gathered.front();
    for (std::size_t i = 1; i != gathered.size(); ++i) {
      if (! isEmpty(gathered[i])) {
        output += gathered[i];
      }
    }
  }
  return output;
}


template<typename _Float, std::size_t _Dimension>
inline
geom::BBox<_Float, _Dimension>
allReduce(geom::BBox<_Float, _Dimension> const& input, MPI_Comm comm)
{
  typedef geom::BBox<_Float, _Dimension> BBox;

  // Perform the reduction to get the result on the root process.
  BBox output = reduce(input, comm);
  // Broadcast the merged bounding box.
  bcast(&output, comm);
  return output;
}


} // namespace mpi
} // namespace stlib
