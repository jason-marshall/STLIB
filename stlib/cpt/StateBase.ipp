// -*- C++ -*-

#if !defined(__cpt_StateBase_ipp__)
#error This file is an implementation detail of the class StateBase.
#endif

namespace stlib
{
namespace cpt
{

template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
displayInformation(std::ostream& out) const
{
  out << "The domain containing the grids is "
      << getDomain() << '\n'
      << "The distance transform will be computed up to "
      << getMaximumDistance() << '\n'
      << "There are " << getNumberOfGrids() << " grids.\n";

  if (hasBRepBeenSet()) {
    out << "The b-rep has been set." << '\n';
    _brep.displayInformation(out);
  }
  else {
    out << "The b-rep has not been set." << '\n';
  }

  if (_hasCptBeenComputed) {
    out << "The closest point transform has been computed." << '\n';
    std::size_t numberKnown = 0;
    Number minimum = std::numeric_limits<Number>::max();
    Number maximum = -std::numeric_limits<Number>::max();
    Number minimumValue, maximumValue;
    // Loop over the grids.
    for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
      numberKnown += _grids[n].countKnownDistances(_maximumDistance);
      _grids[n].computeMinimumAndMaximumDistances(_maximumDistance,
          &minimumValue,
          &maximumValue);
      if (minimumValue < minimum) {
        minimum = minimumValue;
      }
      if (maximumValue > maximum) {
        maximum = maximumValue;
      }
    }
    out << "There are " << numberKnown << " known distances.\n"
        << "The minimum known distance is " << minimum << ".\n"
        << "The maximum known distance is " << maximum << ".\n";
  }
  else {
    out << "The closest point transform has not been computed." << '\n';
  }
}


template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
setParameters(const BBox& domain, const Number maximumDistance)
{
  // Sanity check.
  assert(! isEmpty(domain));
  assert(maximumDistance > 0);

  _domain = domain;
  _maximumDistance = maximumDistance;
}


template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
setLattice(const SizeList& extents, const BBox& domain)
{
  // Sanity check.
  for (std::size_t n = 0; n != N; ++n) {
    assert(extents[n] > 1);
  }
  assert(! isEmpty(domain));

  _lattice = Lattice(extents, domain);
}


template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
insertGrid(const SizeList& extents,
           const IndexList& bases,
           bool useGradientOfDistance,
           bool useClosestPoint,
           bool useClosestFace)
{
  _grids.push_back(Grid());
  _grids.back().rebuild(extents, bases, useGradientOfDistance, useClosestPoint,
                        useClosestFace);
}


template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
insertGrid(const SizeList& extents,
           const IndexList& bases,
           Number* distance,
           Number* gradientOfDistance,
           Number* closestPoint,
           int* closestFace)
{
  _grids.push_back(Grid());
  _grids.back().rebuild(extents, bases, distance, gradientOfDistance,
                        closestPoint, closestFace);
}


template<std::size_t N, typename T>
inline
std::pair<std::size_t, std::size_t>
StateBase<N, T>::
computeClosestPointTransformUsingBBox()
{
  // CONTINUE: Remove the requirement that num_grids > 0.
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the grids.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Compute the closest point transforms.
  return _brep.computeClosestPointUsingBBox(_lattice, &_grids,
         _maximumDistance);
}


template<std::size_t N, typename T>
inline
std::pair<std::size_t, std::size_t>
StateBase<N, T>::
computeClosestPointTransformUsingBruteForce()
{
  // CONTINUE: Remove the requirement that num_grids > 0.
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the grids.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Compute the closest point transforms.
  return _brep.computeClosestPointUsingBruteForce(_lattice, &_grids,
         _maximumDistance);
}


template<std::size_t N, typename T>
inline
std::pair<std::size_t, std::size_t>
StateBase<N, T>::
computeClosestPointTransformUsingTree()
{
  // CONTINUE: Remove the requirement that num_grids > 0.
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the grids.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Make the bounding box tree.
  //CONTINUE HERE;
  assert(false);

  // Compute the closest point transforms for each grid.
  std::pair<std::size_t, std::size_t> counts, c;
  counts.first = 0;
  counts.second = 0;
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    //CONTINUE HERE;
    //c = _grids[n].closestPoint_transform(_lattice, TREE, _maximumDistance);
    counts.first += c.first;
    counts.second += c.second;
  }
  return counts;
}


template<std::size_t N, typename T>
inline
std::pair<std::size_t, std::size_t>
StateBase<N, T>::
computeClosestPointTransformUnsignedUsingBBox()
{
  // CONTINUE: Remove the requirement that num_grids > 0.
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the grids.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Compute the closest point transforms.
  return _brep.computeClosestPointUnsignedUsingBBox(_lattice, &_grids,
         _maximumDistance);
}



template<std::size_t N, typename T>
inline
std::pair<std::size_t, std::size_t>
StateBase<N, T>::
computeClosestPointTransformUnsignedUsingBruteForce()
{
  // CONTINUE: Remove the requirement that num_grids > 0.
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the grids.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Compute the closest point transforms.
  return _brep.computeClosestPointUnsignedUsingBruteForce(_lattice, &_grids,
         _maximumDistance);
}



template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
floodFillAtBoundary(const Number farAway)
{
  // Make sure the cpt has been computed first.
  assert(_hasCptBeenComputed);

  // For each grid.
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    // Flood fill the distance grid.
    _grids[n].floodFill(farAway);
  }
}



template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
floodFillDetermineSign(const Number farAway)
{
  // Make sure the cpt has been computed first.
  assert(_hasCptBeenComputed);

  std::vector<std::size_t> farAwayGrids;
  // For each grid.
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    // Flood fill the distance grid.
    if (! _grids[n].floodFill(farAway)) {
      // Record this grid if there are no known distances.
      farAwayGrids.push_back(n);
    }
  }

  if (farAwayGrids.empty()) {
    return;
  }

  // Determine a Cartesian point that lies in each far away grid.
  std::vector<Point> lowerCorners(farAwayGrids.size());
  for (std::size_t i = 0; i != farAwayGrids.size(); ++i) {
    const IndexList& bases = _grids[farAwayGrids[i]].getRanges().bases();
    for (std::size_t n = 0; n != N; ++n) {
      lowerCorners[i][n] = bases[n];
    }
    _lattice.convertIndexToLocation(&lowerCorners[i]);
  }

  // Determine the signed distance to the lower corners.
  std::vector<Number> distances;
  geom::computeSignedDistance(_brep, lowerCorners.begin(), lowerCorners.end(),
                              std::back_inserter(distances));

  // Set the distances for the far away grids to +- farAway.
  for (std::size_t n = 0; n != farAwayGrids.size(); ++n) {
    std::fill(_grids[farAwayGrids[n]].getDistance().begin(),
              _grids[farAwayGrids[n]].getDistance().end(),
              (distances[n] > 0 ? 1 : -1) * farAway);
  }
}



template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
floodFillUnsigned(const Number farAway)
{
  // Make sure the cpt has been computed first.
  assert(_hasCptBeenComputed);

  // For each grid.
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    // Flood fill the distance grid.
    _grids[n].floodFillUnsigned(farAway);
  }
}



template<std::size_t N, typename T>
inline
bool
StateBase<N, T>::
areGridsValid()
{
  // Check that there are grids.
  if (getNumberOfGrids() == 0) {
    return false;
  }

  const std::size_t FaceIdentifierUpperBound =
    _brep.getFaceIdentifierUpperBound();
  // For each grid.
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    // Check the grid.
    if (! _grids[n].isValid(_lattice, _maximumDistance,
                            FaceIdentifierUpperBound)) {
      std::cerr << "Grid number " << n << " is not valid.\n";
      return false;
    }
  }

  // If we got here, then all the grids are valid.
  return true;
}


template<std::size_t N, typename T>
inline
bool
StateBase<N, T>::
areGridsValidUnsigned()
{
  // Check that there are grids.
  if (getNumberOfGrids() == 0) {
    return false;
  }

  const std::size_t FaceIdentifierUpperBound =
    _brep.getFaceIdentifierUpperBound();
  // For each grid.
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    // Check the grid.
    if (! _grids[n].isValidUnsigned(_lattice, _maximumDistance,
                                    FaceIdentifierUpperBound)) {
      std::cerr << "Grid number " << n << " is not valid.\n";
      return false;
    }
  }

  // If we got here, then all the grids are valid.
  return true;
}

template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
setBRepWithNoClipping
(const std::vector<std::array<Number, N> >& vertices,
 const std::vector<std::array<std::size_t, N> >& faces)
{
  _hasBRepBeenSet = true;
  // Make the BRep.
  _brep.make(vertices, faces);
}

template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
setBRepWithNoClipping(const std::size_t numVertices, const Number* vertices,
                      const std::size_t numFaces, const int* faces)
{
  std::vector<std::array<Number, N> > v(numVertices);
  for (std::size_t i = 0; i != v.size(); ++i) {
    for (std::size_t n = 0; n != N; ++n) {
      v[i][n] = *vertices++;
    }
  }
  std::vector<std::array<std::size_t, N> > f(numFaces);
  for (std::size_t i = 0; i != f.size(); ++i) {
    for (std::size_t n = 0; n != N; ++n) {
      f[i][n] = *faces++;
    }
  }
  setBRepWithNoClipping(v, f);
}

template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
setBRep(const std::vector<std::array<Number, N> >& vertices,
        const std::vector<std::array<std::size_t, N> >& faces)
{
  _hasBRepBeenSet = true;
  // Check that the cartesian domain has been set.
  assert(! isEmpty(_domain) && _maximumDistance > 0);
  // Make the BRep.
  _brep.make(vertices, faces, getDomain(), _maximumDistance);
}

template<std::size_t N, typename T>
inline
void
StateBase<N, T>::
setBRep(const std::size_t numVertices, const Number* vertices,
        const std::size_t numFaces, const int* faces)
{
  std::vector<std::array<Number, N> > v(numVertices);
  for (std::size_t i = 0; i != v.size(); ++i) {
    for (std::size_t n = 0; n != N; ++n) {
      v[i][n] = *vertices++;
    }
  }
  std::vector<std::array<std::size_t, N> > f(numFaces);
  for (std::size_t i = 0; i != f.size(); ++i) {
    for (std::size_t n = 0; n != N; ++n) {
      f[i][n] = *faces++;
    }
  }
  setBRep(v, f);
}

} // namespace cpt
}
