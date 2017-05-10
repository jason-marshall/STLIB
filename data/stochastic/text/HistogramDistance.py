# HistogramDistance.py

import sys
sys.path.append('../../../applications/stochastic/state')
from Histogram import Histogram, histogramDistance

if __name__ == '__main__':
  errorMessage = \
  """Usage:
  python HistogramDistance.py"""

  if len(sys.argv) != 1:
    print errorMessage
    raise AssertionError, "Wrong number of command line arguments.  Exiting..."


  x = Histogram()
  x.read(sys.stdin)
  y = Histogram()
  y.read(sys.stdin)
  print histogramDistance(x, y)
