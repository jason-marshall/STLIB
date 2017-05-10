// -*- C++ -*-

/*!
  \file stochastic/HistogramsAveragePackedDouble.h
  \brief Two packed sets of histograms.
*/

#if !defined(__stochastic_HistogramsAveragePackedDouble_h__)
#define __stochastic_HistogramsAveragePackedDouble_h__

#include "stlib/stochastic/HistogramsAveragePacked.h"

namespace stlib
{
namespace stochastic
{

//! Two packed sets of histograms.
/*!
  Call initialize before recording each trajectory. The sum of the first
  histogram is not less than the sum of the second.
*/
class HistogramsAveragePackedDouble
{
private:

  //
  // Member data.
  //

  // The first packed set of histograms.
  HistogramsAveragePacked _first;
  // The second packed set of histograms.
  HistogramsAveragePacked _second;
  // The current set of histograms.
  HistogramsAveragePacked* _histograms;

  //
  // Not implemented.
  //

private:

  //! Default constructor not implemented.
  HistogramsAveragePackedDouble();
  //! Copy constructor not implemented.
  HistogramsAveragePackedDouble(const HistogramsAveragePackedDouble&);
  //! Assignment operator not implemented.
  HistogramsAveragePackedDouble&
  operator=(const HistogramsAveragePackedDouble&);


  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the number of species and the number of bins.
  HistogramsAveragePackedDouble(const std::size_t numberOfSpecies,
                                const std::size_t numberOfBins) :
    _first(numberOfSpecies, numberOfBins),
    _second(numberOfSpecies, numberOfBins),
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
  operator()(const std::size_t species) const
  {
    return (*_histograms)(species);
  }

  //! Return the first set of packed histograms.
  const HistogramsAveragePacked&
  first() const
  {
    return _first;
  }

  //! Return the second set of packed histograms.
  const HistogramsAveragePacked&
  second() const
  {
    return _second;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Prepare for recording a trajectory by selecting a set of histograms and incrementing the number of trajectories.
  void
  initialize()
  {
    if (_first.getNumberOfTrajectories() > _second.getNumberOfTrajectories()) {
      _histograms = &_second;
    }
    else {
      _histograms = &_first;
    }
    _histograms->incrementNumberOfTrajectories();
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    for (std::size_t species = 0; species != _first.numberOfSpecies();
         ++species) {
      stochastic::synchronize(&_first(species),
                              &_second(species));
    }
  }

  //! Return the specified histogram.
  HistogramReference&
  operator()(const std::size_t species)
  {
    return (*_histograms)(species);
  }

  //@}
};

//! Write the two sets of histograms in ascii format.
/*!
  \relates HistogramsAveragePackedDouble

  \note You must call synchronize() on \c x before calling this function.

  For each species: write a histogram with two bin arrays.
*/
inline
std::ostream&
operator<<(std::ostream& out, const HistogramsAveragePackedDouble& x)
{
  for (std::size_t species = 0; species != x.second().numberOfSpecies();
       ++species) {
    // Write the lower bound, width, and the bins for the first histogram.
    const HistogramReference& h1 = x.first()(species);
    out << h1;
    // Write the bins for the second histogram.
    const HistogramReference& h2 = x.second()(species);
    assert(h1.getLowerBound() == h2.getLowerBound() &&
           h1.getWidth() == h2.getWidth());
    std::copy(h2.begin(), h2.end(), std::ostream_iterator<double>(out, " "));
    out << '\n';
  }
  return out;
}

} // namespace stochastic
}

#endif
