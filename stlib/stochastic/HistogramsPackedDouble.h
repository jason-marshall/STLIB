// -*- C++ -*-

/*!
  \file stochastic/HistogramsPackedDouble.h
  \brief A reaction.
*/

#if !defined(__stochastic_HistogramsPackedDouble_h__)
#define __stochastic_HistogramsPackedDouble_h__

#include "stlib/stochastic/HistogramsPacked.h"

namespace stlib
{
namespace stochastic
{

//! Two packed sets of histograms.
/*!
  Call initialize before recording each trajectory. The sum of the first
  histogram is not less than the sum of the second.
*/
class HistogramsPackedDouble
{
private:

  //
  // Member data.
  //

  // The first packed set of histograms.
  HistogramsPacked _first;
  // The second packed set of histograms.
  HistogramsPacked _second;
  // The current set of histograms.
  HistogramsPacked* _histograms;

  //
  // Not implemented.
  //

private:

  //! Default constructor not implemented.
  HistogramsPackedDouble();
  //! Copy constructor not implemented.
  HistogramsPackedDouble(const HistogramsPackedDouble&);
  //! Assignment operator not implemented.
  HistogramsPackedDouble&
  operator=(const HistogramsPackedDouble&);


  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the number of frames, the number of species, and the number of bins.
  HistogramsPackedDouble(const std::size_t numberOfFrames,
                         const std::size_t numberOfSpecies,
                         const std::size_t numberOfBins) :
    _first(numberOfFrames, numberOfSpecies, numberOfBins),
    _second(numberOfFrames, numberOfSpecies, numberOfBins),
    _histograms(&_first)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the specified histogram.
  const HistogramReference&
  operator()(const std::size_t frame, const std::size_t species) const
  {
    return (*_histograms)(frame, species);
  }

  //! Return the first set of packed histograms.
  const HistogramsPacked&
  first() const
  {
    return _first;
  }

  //! Return the second set of packed histograms.
  const HistogramsPacked&
  second() const
  {
    return _second;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Prepare for recording a trajectory by selecting a set of histograms and incrementing the sum.
  void
  initialize()
  {
    if (_first.getSum() > _second.getSum()) {
      _histograms = &_second;
    }
    else {
      _histograms = &_first;
    }
    _histograms->incrementSum();
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    for (std::size_t frame = 0; frame != _first.numberOfFrames(); ++frame) {
      for (std::size_t species = 0; species != _first.numberOfSpecies();
           ++species) {
        stochastic::synchronize(&_first(frame, species),
                                &_second(frame, species));
      }
    }
  }

  //! Return the specified histogram.
  HistogramReference&
  operator()(const std::size_t frame, const std::size_t species)
  {
    return (*_histograms)(frame, species);
  }

  //@}
};

//! Write the two sets of histograms in ascii format.
/*!
  \relates HistogramsPackedDouble

  \note You must call synchronize() on \c x before calling this function.

  For each frame: for each species: write a histogram with two bin arrays.
*/
inline
std::ostream&
operator<<(std::ostream& out, const HistogramsPackedDouble& x)
{
  for (std::size_t frame = 0; frame != x.first().numberOfFrames(); ++frame) {
    for (std::size_t species = 0; species != x.second().numberOfSpecies();
         ++species) {
      // Write the lower bound, width, and the bins for the first histogram.
      const HistogramReference& h1 = x.first()(frame, species);
      out << h1;
      // Write the bins for the second histogram.
      const HistogramReference& h2 = x.second()(frame, species);
      assert(h1.getLowerBound() == h2.getLowerBound() &&
             h1.getWidth() == h2.getWidth());
      std::copy(h2.begin(), h2.end(), std::ostream_iterator<double>(out, " "));
      out << '\n';
    }
  }
  return out;
}

} // namespace stochastic
}

#endif
