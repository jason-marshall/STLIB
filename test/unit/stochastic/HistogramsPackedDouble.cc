// -*- C++ -*-

#include "stlib/stochastic/HistogramsPackedDouble.h"

using namespace stlib;

int
main()
{
  typedef stochastic::HistogramsPackedDouble HistogramsPackedDouble;
  typedef stochastic::HistogramReference HistogramReference;

  const std::size_t NumberOfFrames = 3;
  const std::size_t NumberOfSpecies = 2;

  {
    // Empty.
    const std::size_t Size = 32;
    HistogramsPackedDouble histograms(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        {
          const HistogramReference& x = histograms.first()(frame, species);
          assert(x.size() == Size);
          assert(x.computeSum() == 0);
          assert(x.getLowerBound() == 0);
          assert(x.getWidth() == 1);
          assert(x.getUpperBound() == Size);
          for (std::size_t i = 0; i != x.size(); ++i) {
            assert(x[i] == 0);
          }
        }
        {
          const HistogramReference& x = histograms.second()(frame, species);
          assert(x.size() == Size);
          assert(x.computeSum() == 0);
          assert(x.getLowerBound() == 0);
          assert(x.getWidth() == 1);
          assert(x.getUpperBound() == Size);
          for (std::size_t i = 0; i != x.size(); ++i) {
            assert(x[i] == 0);
          }
        }
      }
    }
  }
  {
    // Insert within range.
    const std::size_t Size = 32;
    HistogramsPackedDouble histograms(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        for (std::size_t i = 0; i != 2; ++i) {
          histograms.initialize();
          HistogramReference& x = histograms(frame, species);
          for (std::size_t j = 0; j != Size; ++j) {
            x.accumulate(j, 1);
          }
        }
        {
          const HistogramReference& x = histograms.first()(frame, species);
          assert(x.size() == Size);
          assert(x.computeSum() == Size);
          assert(x.getLowerBound() == 0);
          assert(x.getWidth() == 1);
          assert(x.getUpperBound() == Size);
          for (std::size_t i = 0; i != x.size(); ++i) {
            assert(x[i] == 1);
          }
        }
        {
          const HistogramReference& x = histograms.second()(frame, species);
          assert(x.size() == Size);
          assert(x.computeSum() == Size);
          assert(x.getLowerBound() == 0);
          assert(x.getWidth() == 1);
          assert(x.getUpperBound() == Size);
          for (std::size_t i = 0; i != x.size(); ++i) {
            assert(x[i] == 1);
          }
        }
      }
    }
  }

  return 0;
}
