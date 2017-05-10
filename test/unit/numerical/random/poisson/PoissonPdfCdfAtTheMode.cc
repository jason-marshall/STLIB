// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonPdfCdfAtTheMode.h"

#include <iostream>

using namespace stlib;

int
main()
{
  numerical::PoissonPdfCdfAtTheMode<> pdfCdf(0, 32, 100);
  numerical::PoissonPdf<> pdf;
  numerical::PoissonCdf<> cdf;
  double x, p, c, deviation, pdfDeviation = 0, cdfDeviation = 0,
                             pdfX = -1, cdfX = -1;
  for (int i = 0; i != 32 * 1000; ++i) {
    x = i / 1000.0;
    pdfCdf.evaluate(x, &p, &c);
    deviation = std::abs(p - pdf(x, int(x))) / pdf(x, int(x));
    if (deviation > pdfDeviation) {
      pdfDeviation = deviation;
      pdfX = x;
    }
    deviation = std::abs(c - cdf(x, int(x))) / cdf(x, int(x));
    if (deviation > cdfDeviation) {
      cdfDeviation = deviation;
      cdfX = x;
    }
  }
  std::cout << "Maximum relative deviation in the PDF of "
            << pdfDeviation << " at " << pdfX << "\n"
            << "Maximum relative deviation in the CDF of "
            << cdfDeviation << " at " << cdfX << "\n";

  return 0;
}
