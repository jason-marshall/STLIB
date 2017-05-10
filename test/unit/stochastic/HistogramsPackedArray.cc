// -*- C++ -*-

#include "stlib/stochastic/HistogramsPackedArray.h"
#include "stlib/numerical/random/uniform.h"

using namespace stlib;

int
main()
{
  typedef stochastic::HistogramsPackedArray HistogramsPackedArray;
  typedef stochastic::HistogramReference HistogramReference;


  {
    // Empty.
    const std::size_t NumberOfFrames = 3;
    const std::size_t NumberOfSpecies = 2;
    const std::size_t Size = 32;
    const std::size_t Multiplicity = 4;
    HistogramsPackedArray histograms(NumberOfFrames, NumberOfSpecies, Size,
                                     Multiplicity);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        for (std::size_t i = 0; i != Multiplicity; ++i) {
          const HistogramReference& x = histograms.get(i)(frame, species);
          assert(x.size() == Size);
          assert(x.computeSum() == 0);
          assert(x.getLowerBound() == 0);
          assert(x.getWidth() == 1);
          assert(x.getUpperBound() == Size);
          for (std::size_t j = 0; j != x.size(); ++j) {
            assert(x[j] == 0);
          }
        }
      }
    }
  }
  {
    // Insert within range.
    const std::size_t NumberOfFrames = 3;
    const std::size_t NumberOfSpecies = 2;
    const std::size_t Size = 32;
    const std::size_t Multiplicity = 4;
    HistogramsPackedArray histograms(NumberOfFrames, NumberOfSpecies, Size,
                                     Multiplicity);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        for (std::size_t i = 0; i != Multiplicity; ++i) {
          histograms.initialize();
          HistogramReference& x = histograms(frame, species);
          for (std::size_t j = 0; j != Size; ++j) {
            x.accumulate(j, 1);
          }
        }
        for (std::size_t i = 0; i != Multiplicity; ++i) {
          const HistogramReference& x = histograms.get(i)(frame, species);
          assert(x.size() == Size);
          assert(x.computeSum() == Size);
          assert(x.getLowerBound() == 0);
          assert(x.getWidth() == 1);
          assert(x.getUpperBound() == Size);
          for (std::size_t j = 0; j != x.size(); ++j) {
            assert(x[j] == 1);
          }
        }
      }
    }
  }
  {
    // Synchronize.
    const std::size_t NumberOfFrames = 1;
    const std::size_t NumberOfSpecies = 1;
    const std::size_t Size = 4;
    const std::size_t Multiplicity = 4;
    numerical::DiscreteUniformGeneratorMt19937 generator;
    for (std::size_t count = 0; count != 100; ++count) {
      HistogramsPackedArray histograms(NumberOfFrames, NumberOfSpecies, Size,
                                       Multiplicity);
      for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
        for (std::size_t species = 0; species != NumberOfSpecies;
             ++species) {
          for (std::size_t i = 0; i != Multiplicity; ++i) {
            histograms.initialize();
            HistogramReference& x = histograms(frame, species);
            x.accumulate(generator() % 1024, 1);
          }
        }
      }
      histograms.synchronize();
    }
  }

  return 0;
}
