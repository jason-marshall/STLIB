// -*- C++ -*-

#if !defined(__sfc_BlockCode_tcc__)
#error This file is an implementation detail of BlockCode.
#endif

namespace stlib
{
namespace sfc
{


//--------------------------------------------------------------------------
// Constructors etc.


template<typename _Traits>
inline
BlockCode<_Traits>::
BlockCode() :
  Base(),
  _order(),
  _levelBits(0),
  _levelMask(0),
  _locationMasks(),
  _increments()
{
}


template<typename _Traits>
inline
BlockCode<_Traits>::
BlockCode(const Point& lowerCorner, const Point& lengths,
          const std::size_t numLevels) :
  Base(lowerCorner, lengths, numLevels),
  _order(),
  _levelBits(numLevels ? numerical::highestBitPosition(numLevels) + 1 : 0),
  _levelMask(_levelBits ? (Code(1) << _levelBits) - 1 : 0),
  _locationMasks(),
  _increments()
{
  _initialize();
}


template<typename _Traits>
inline
BlockCode<_Traits>::
BlockCode(const BBox& tbb, const Float minCellLength) :
  // Determine the lower corner, lengths, and number of levels with the base
  // constructor.
  Base(tbb, minCellLength)
{
  // Call the other constructor.
  *this = BlockCode(Base::_lowerCorner, Base::_lengths, Base::_numLevels);
}


template<typename _Traits>
inline
void
BlockCode<_Traits>::
_initialize()
{
  // Set the location masks.
  {
    Code mask = ((Code(1) << Dimension) - 1) <<
                           (_levelBits + Dimension * (Base::_numLevels - 1));
    std::fill(_locationMasks.begin(), _locationMasks.end(), Code(0));
    for (std::size_t i = 1; i <= Base::_numLevels; ++i) {
      _locationMasks[i] = mask;
      mask >>= Dimension;
    }
  }
  // Set the increment bits.
  {
    Code bit = Code(1) <<
                          (_levelBits + Dimension * Base::_numLevels);
    for (std::size_t i = 0; i != _increments.size(); ++i) {
      _increments[i] = bit;
      bit >>= Dimension;
    }
  }
}


template<typename _Traits>
inline
void
BlockCode<_Traits>::
setNumLevels(const std::size_t numLevels)
{
  Base::setNumLevels(numLevels);
  _levelBits = numLevels ? numerical::highestBitPosition(numLevels) + 1 : 0;
  _levelMask = _levelBits ? (Code(1) << _levelBits) - 1 : 0;
  _initialize();
}


//--------------------------------------------------------------------------
// Manipulate block codes.


template<typename _Traits>
inline
typename BlockCode<_Traits>::Code
BlockCode<_Traits>::
code(const Point& p) const
{
  // First build the location code without the level information. Next encode
  // the finest level.
  return _order.code(Base::coordinates(p), Base::numLevels()) << _levelBits |
         Base::_numLevels;
}


template<typename _Traits>
inline
bool
BlockCode<_Traits>::
isValid(const Code code) const
{
  // Check that bits more significant than those used are not set.
  if (code >= Code(1) << (_levelBits + Dimension * Base::_numLevels)) {
    return false;
  }
  // Check that the level is in the valid range.
  const std::size_t lev = level(code);
  if (lev > Base::_numLevels) {
    return false;
  }
  // Check that location bits are not set past the indicated level.
  for (std::size_t i = lev + 1; i <= Base::_numLevels; ++i) {
    if (code & _locationMasks[i]) {
      return false;
    }
  }
  return true;
}


template<typename _Traits>
inline
typename BlockCode<_Traits>::Code
BlockCode<_Traits>::
parent(const Code code) const
{
  const std::size_t lev = level(code);
  assert(lev != 0);
  // Mask out the location bits at the current level and then decrement the
  // level.
  return (code & ~_locationMasks[lev]) - 1;
}


template<typename _Traits>
inline
typename BlockCode<_Traits>::Code
BlockCode<_Traits>::
atLevel(Code code, const std::size_t n) const
{
#ifdef STLIB_DEBUG
  assert(n <= Base::numLevels());
#endif
  const std::size_t shift = (Base::numLevels() - n) * Dimension + _levelBits;
  // Right shift to erase the level number and bits for higher levels,
  // left shift to restore the location, and then add the new level number.
  return (code >> shift << shift) + n;
}


template<typename _Traits>
inline
bool
BlockCode<_Traits>::
operator==(const BlockCode& other) const
{
  return Base::operator==(other) &&
         _levelBits == other._levelBits &&
         _levelMask == other._levelMask &&
         _locationMasks == other._locationMasks &&
         _increments == other._increments;
}


template<typename _Traits, typename _Object>
inline
void
sort(BlockCode<_Traits> const& blockCode,
     std::vector<_Object>* objects,
     std::vector<typename _Traits::Code>* objectCodes)
{
  typedef typename _Traits::Code Code;
  typedef typename _Traits::BBox BBox;

  // A vector of code/index pairs.
  std::vector<std::pair<Code, std::size_t> >
    codeIndexPairs(objects->size());
  // Calculate the codes.
  for (std::size_t i = 0; i != codeIndexPairs.size(); ++i) {
    codeIndexPairs[i].first =
      blockCode.code(centroid(geom::specificBBox<BBox>((*objects)[i])));
    codeIndexPairs[i].second = i;
  }
  // Sort according to the codes.
  lorg::sort(&codeIndexPairs, blockCode.numBits());
  // Set the sorted codes and objects.
  objectCodes->resize(codeIndexPairs.size());
  {
    std::vector<_Object> obj(codeIndexPairs.size());
    for (std::size_t i = 0; i != codeIndexPairs.size(); ++i) {
      (*objectCodes)[i] = codeIndexPairs[i].first;
      obj[i] = (*objects)[codeIndexPairs[i].second];
    }
    objects->swap(obj);
  }
}


template<typename _Traits>
inline
void
coarsen(BlockCode<_Traits> const& blockCode,
        std::vector<typename _Traits::Code> const& objectCodes,
        std::vector<typename _Traits::Code>* cellCodes,
        std::size_t const maxObjectsPerCell)
{
  typedef typename _Traits::Code Code;

  // Initially, the objects codes are all at the highest level of refinement.

  cellCodes->clear();
  
  // The first admissible code to append.
  Code firstAdmissible = 0;
  for (std::size_t i = 0; i != objectCodes.size(); /*advance inside*/) {
    // Start with a cell at the highest level of refinement 
    Code code = objectCodes[i];
    std::size_t end = i + 1;
    // Find the contents of the cell.
    for ( ; end != objectCodes.size() && objectCodes[end] == code; ++end) {
    }
    // If we exceed the maximum objects in a cell here, we will insert a cell
    // at the highest level.
    if (end - i <= maxObjectsPerCell) {
      std::size_t const indexOver = std::min(i + maxObjectsPerCell + 1,
                                             objectCodes.size());
      // Iterate to find an appropriate level.
      while (blockCode.level(code) != 0) {
        // Try moving to a lower level.
        Code const parent = blockCode.parent(code);
        // If the parent intersects the previous cell, we can't coarsen.
        if (parent < firstAdmissible) {
          break;
        }
        // Note that the object codes are at the highest level of refinement
        // so we don't need to convert to a location here.
        Code const next = blockCode.next(parent);
        std::size_t parentEnd = end;
        for ( ; parentEnd != indexOver && objectCodes[parentEnd] < next;
              ++parentEnd) {
        }
        // If the cell would contain too many objects, we can't coarsen.
        if (parentEnd - i > maxObjectsPerCell) {
          break;
        }
        // Move to a lower level.
        code = parent;
        end = parentEnd;
      }
    }
    i = end;
    // Add the cell.
    cellCodes->push_back(code);
    // The codes that we add after this may not intersect the code we just
    // added.
    firstAdmissible = blockCode.location(blockCode.next(code));
  }
  // The sequence is terminated with the guard code.
  cellCodes->push_back(Code(_Traits::GuardCode));
}


} // namespace sfc
}
