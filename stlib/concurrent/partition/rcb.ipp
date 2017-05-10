// -*- C++ -*-

#if !defined(__partition_rcb_ipp__)
#error This file is an implementation detail of rcb.
#endif

namespace stlib
{
namespace concurrent
{

template<std::size_t N, typename IDType>
inline
void
rcbSplit(IDType** partitionBegin, IDType** partitionEnd,
         const std::array<double, N>** recordsBegin,
         const std::array<double, N>** recordsEnd)
{
  typedef std::array<double, N> Point;

  // If there is only a single processor, there is no need to split.
  const std::size_t numProcessors = partitionEnd - partitionBegin;
  if (numProcessors == 1) {
    return;
  }

  //
  // Determine the number of processors and records in the left and
  // right branches.
  //
  const std::size_t numProcessorsLeft = numProcessors / 2;
  IDType** partitionMid = partitionBegin + numProcessorsLeft;
  const std::size_t numRecords = recordsEnd - recordsBegin;
  // Use the double type to avoid overflow.
  const std::size_t numRecordsLeft =
    std::size_t(double(numRecords * numProcessorsLeft) / numProcessors);
  *partitionMid = *partitionBegin + numRecordsLeft;

  //
  // Determine the splitting dimension.
  //
  typedef geom::BBox<double, N> BBox;
  BBox const bbox =
    geom::specificBBox<BBox>(ads::constructIndirectIterator(recordsBegin),
                             ads::constructIndirectIterator(recordsEnd));
  Point extents = bbox.upper - bbox.lower;
  const std::size_t splitDimension =
    std::max_element(extents.begin(), extents.end()) - extents.begin();

  //
  // The comparison functor for splitting the records.
  //
  // Dereference the pointer.
  typedef ads::Dereference<const Point*> Deref;
  Deref deref;
  // Index the point.
  typedef ads::IndexConstObject<Point> Index;
  Index ind;
  // Index in the splitting dimension.
  typedef std::binder2nd<Index> IndexSD;
  IndexSD indsd(ind, splitDimension);
  // Dereference; then index in the splitting dimension.
  typedef ads::unary_compose_unary_unary<IndexSD, Deref> ISDD;
  ISDD isdd(indsd, deref);
  // Compare two handles to points by their splitting dimension coordinate.
  typedef ads::binary_compose_binary_unary<std::less<double>, ISDD, ISDD>
  Comp;
  std::less<double> lessThan;
  Comp comp(lessThan, isdd, isdd);

  //
  // Split the records.
  //
  const Point** recordsMid = recordsBegin + numRecordsLeft;
  std::nth_element(recordsBegin, recordsMid, recordsEnd, comp);

  //
  // Recurse.
  //
  rcbSplit(partitionBegin, partitionMid, recordsBegin, recordsMid);
  rcbSplit(partitionMid, partitionEnd, recordsMid, recordsEnd);
}


template<std::size_t N, typename IDType>
inline
void
rcb(const std::size_t numProcessors,
    std::vector<IDType>* identifiers, std::vector<IDType*>* idPartition,
    const std::vector<std::array<double, N> >& positions)
{
  typedef std::array<double, N> Point;

  assert(numProcessors >= 1);
  assert(identifiers->size() == positions.size());
  assert(idPartition->size() == numProcessors + 1);

  // Assign the first and last partition pointers.
  (*idPartition)[0] = &(*identifiers)[0];
  (*idPartition)[numProcessors] = (*idPartition)[0] + identifiers->size();

  // Make an array of pointers to the positions.
  std::vector<const Point*> posPtrs(positions.size());
  for (std::size_t i = 0; i != posPtrs.size(); ++i) {
    posPtrs[i] = &positions[i];
  }

  // Determine the partition.
  rcbSplit(&(*idPartition)[0], &(*idPartition)[numProcessors],
           &posPtrs[0], &posPtrs[0] + posPtrs.size());

  // Re-arrange the identifiers.
  std::vector<IDType> tmp(*identifiers);
  for (std::size_t i = 0; i != identifiers->size(); ++i) {
    (*identifiers)[i] = tmp[posPtrs[i] - &positions[0]];
  }
}

} // namespace concurrent
}
