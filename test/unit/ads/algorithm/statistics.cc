// -*- C++ -*-

#include "stlib/ads/algorithm/statistics.h"

#include "stlib/ads/functor/linear.h"

#include <iostream>

#include <cassert>


using namespace stlib;

int
main()
{
  {
    const int size = 4;
    int a[size] = {3, 2, 1, 0};
    int minimum, maximum, mean;

    ads::computeMinimumMaximumAndMean(&a[0], a + size,
                                      &minimum, &maximum, &mean);
    assert(minimum == 0);
    assert(maximum == 3);
    assert(mean == 1);

#if 0
    // CONTINUE: use a TransformIterator.
    ads::computeMinimumMaximumAndMean(&a[0], a + size,
                                      &minimum, &maximum, &mean,
                                      ads::UnaryLinear<int>(2, 3));
    assert(minimum == 3);
    assert(maximum == 9);
    assert(mean == 6);
#endif
  }

  {
    const double Data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int Size = sizeof(Data) / sizeof(double);
    const double ActualMean = 9.0 / 2.0;
    const double ActualVariance = 55.0 / 6.0;
    double mean, variance;
    ads::computeMeanAndVariance(Data, Data + Size, &mean, &variance);
    std::cout << "Computed mean = " << mean
              << ", actual mean = " << ActualMean
              << ", difference = " << mean - ActualMean << "\n"
              << "Computed variance = " << variance
              << ", actual variance = " << ActualVariance
              << ", difference = " << variance - ActualVariance << "\n\n";

    const double ActualAbsoluteDeviation = 5. / 2.;
    const double ActualSkew = 0;
    const double ActualCurtosis = 7911. / 5500. - 3.;
    double absoluteDeviation, skew, curtosis;
    ads::computeMeanAbsoluteDeviationVarianceSkewAndCurtosis
    (Data, Data + Size, &mean, &absoluteDeviation, &variance, &skew,
     &curtosis);
    std::cout << "Computed mean = " << mean
              << ", actual mean = " << ActualMean
              << ", difference = " << mean - ActualMean << "\n"
              << "Computed variance = " << variance
              << ", actual variance = " << ActualVariance
              << ", difference = " << variance - ActualVariance << "\n"
              << "Computed absolute deviation = " << absoluteDeviation
              << ", actual absolute deviation = " << ActualAbsoluteDeviation
              << ", difference = "
              << absoluteDeviation - ActualAbsoluteDeviation << "\n"
              << "Computed skew = " << skew
              << ", actual skew = " << ActualSkew
              << ", difference = " << skew - ActualSkew << "\n"
              << "Computed curtosis = " << curtosis
              << ", actual curtosis = " << ActualCurtosis
              << ", difference = " << curtosis - ActualCurtosis << "\n\n";
  }
  {
    const double Data[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9};
    const int Size = sizeof(Data) / sizeof(double);
    const double ActualMean = 111111111.1;
    const double ActualVariance = 98516024444693852.1;
    double mean, variance;
    ads::computeMeanAndVariance(Data, Data + Size, &mean, &variance);
    std::cout << "Computed mean = " << mean
              << ", actual mean = " << ActualMean
              << ", difference = " << mean - ActualMean << "\n"
              << "Computed variance = " << variance
              << ", actual variance = " << ActualVariance
              << ", difference = " << variance - ActualVariance << "\n\n";

    const double ActualAbsoluteDeviation = 8888888889. / 50.;
    const double ActualSkew = 704545423434655506. *
                              std::sqrt(6. / 182437082304988615.) / 1806127115.;
    const double ActualCurtosis = 653377539998302031538196281. /
                                  653377539998302031538196281. - 3.;
    double absoluteDeviation, skew, curtosis;
    ads::computeMeanAbsoluteDeviationVarianceSkewAndCurtosis
    (Data, Data + Size, &mean, &absoluteDeviation, &variance, &skew,
     &curtosis);
    std::cout << "Computed mean = " << mean
              << ", actual mean = " << ActualMean
              << ", difference = " << mean - ActualMean << "\n"
              << "Computed variance = " << variance
              << ", actual variance = " << ActualVariance
              << ", difference = " << variance - ActualVariance << "\n"
              << "Computed absolute deviation = " << absoluteDeviation
              << ", actual absolute deviation = " << ActualAbsoluteDeviation
              << ", difference = "
              << absoluteDeviation - ActualAbsoluteDeviation << "\n"
              << "Computed skew = " << skew
              << ", actual skew = " << ActualSkew
              << ", difference = " << skew - ActualSkew << "\n"
              << "Computed curtosis = " << curtosis
              << ", actual curtosis = " << ActualCurtosis
              << ", difference = " << curtosis - ActualCurtosis << "\n\n";
  }
}
