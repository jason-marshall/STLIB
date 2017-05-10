// -*- C++ -*-

#include "stlib/numerical/random/poisson/PoissonPdfAtTheMode.h"

#include <iostream>

using namespace stlib;

int
main()
{
  numerical::PoissonPdfAtTheMode<> pdfAtTheMode(0, 32, 100);
  numerical::PoissonPdf<> pdf;
  double x, deviation, maxDeviation = 0, maxX = -1;
  for (int i = 0; i != 32 * 1000; ++i) {
    x = i / 1000.0;
    deviation = std::abs(pdfAtTheMode(x) - pdf(x, int(x))) / pdf(x, int(x));
    if (deviation > maxDeviation) {
      maxDeviation = deviation;
      maxX = x;
    }
  }
  std::cout << "Maximum relative deviation of " << maxDeviation << " at "
            << maxX << "\n";

  return 0;
}
